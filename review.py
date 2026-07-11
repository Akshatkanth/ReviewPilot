import argparse
import json
import subprocess
import sys
import tempfile

from dotenv import load_dotenv
import os
from github import Github, Auth, GithubException
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List

# Load environment variables from .env
load_dotenv()


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class SecurityIssue(BaseModel):
    """A single security finding judged genuinely concerning by the LLM."""
    file: str = Field(description="Filename where the issue was found")
    line_number: int = Field(description="Line number of the issue in the file")
    description: str = Field(description="Clear description of the security issue")
    why_it_matters: str = Field(
        description="1-2 sentences explaining the real-world risk or impact"
    )


class SecurityAgentOutput(BaseModel):
    """Structured output from the Security Agent LLM call."""
    overall_severity: str = Field(
        description=(
            "Overall severity of real security concerns across all files. "
            "Must be one of: 'none', 'low', 'medium', 'high', 'critical'."
        )
    )
    real_issues: List[SecurityIssue] = Field(
        description="Bandit findings the LLM judges as genuinely concerning given the code context"
    )
    dismissed_noise: List[str] = Field(
        description=(
            "Brief descriptions of bandit findings dismissed as false positives, "
            "test fixtures, or non-issues given the context."
        )
    )
    reasoning: str = Field(
        description="2-4 sentences summarising the overall security judgment and key rationale"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="PR Review Agent — fetch PR metadata")
    parser.add_argument("--repo", required=True, help="Repository in 'owner/reponame' format")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    return parser.parse_args()


def test_llm_connection():
    """Send a hardcoded prompt to Groq and print the response + token usage."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment. Check your .env file.")
        sys.exit(1)

    print(f"\n{'='*50}")
    print("  LLM CONNECTION TEST")
    print(f"{'='*50}")

    # ChatGroq is a LangChain wrapper around the Groq cloud API.
    # It handles auth, request formatting, and response parsing for us.
    # "llama-3.3-70b-versatile" = Meta's Llama 3.3 model, 70 billion parameters,
    # hosted on Groq's ultra-fast inference hardware (LPU chips).
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
    )

    prompt = "In one sentence, explain what a pull request is."
    print(f"  Prompt  : {prompt}")

    # llm.invoke() sends the message list to Groq and returns an AIMessage object.
    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"  Response: {response.content}")

    # Token usage is attached to response.response_metadata by langchain-groq
    usage = response.response_metadata.get("token_usage", {})
    if usage:
        print(f"  Tokens  : {usage.get('prompt_tokens', '?')} prompt "
              f"+ {usage.get('completion_tokens', '?')} completion "
              f"= {usage.get('total_tokens', '?')} total")
    else:
        print("  Tokens  : (usage metadata not available)")

    print(f"{'='*50}\n")


def run_bandit_scan(python_files, repo, pr):
    """
    Runs bandit static security analysis on each changed Python file.

    Strategy: fetch the full file content at the PR's head commit from GitHub
    (not just the patch fragment), write it to a temporary .py file so bandit
    has a complete, valid Python source to analyse, run bandit via subprocess
    with JSON output, parse and print findings, then delete the temp file.

    Args:
        python_files: list of PullRequestFile objects from pr.get_files()
        repo:         PyGithub Repository object (needed for get_contents)
        pr:           PyGithub PullRequest object (needed for head SHA)
    """
    head_sha = pr.head.sha

    print(f"\n{'='*50}")
    print("  BANDIT SECURITY SCAN")
    print(f"{'='*50}\n")

    # Collected results returned to the caller for use by run_security_agent()
    bandit_results = []   # list of dicts: {filename, issues: [...]}

    for pr_file in python_files:
        filename = pr_file.filename
        print(f"  Scanning: {filename}")

        # --- Fetch full file content at the PR head commit ---
        # bandit needs syntactically valid, complete Python source.
        # A raw diff/patch fragment is not valid Python on its own.
        if pr_file.status == "removed":
            print("    (file was deleted in this PR — skipping bandit scan)\n")
            continue

        try:
            contents = repo.get_contents(filename, ref=head_sha)
            file_bytes = contents.decoded_content  # bytes
        except Exception as e:
            print(f"    Warning: could not fetch full content for {filename}: {e}")
            print("    Falling back to patch fragment — results may be incomplete.")
            file_bytes = (pr_file.patch or "").encode("utf-8")

        # --- Write to a named temp file (.py extension required by bandit) ---
        tmp = tempfile.NamedTemporaryFile(
            suffix=".py",
            delete=False,   # we'll delete manually after scanning
            mode="wb",
        )
        try:
            tmp.write(file_bytes)
            tmp.flush()
            tmp.close()

            # --- Run bandit via subprocess with JSON output ---
            result = subprocess.run(
                ["bandit", "--format", "json", "--quiet", tmp.name],
                capture_output=True,
                text=True,
            )

            # bandit exits 0 (no issues), 1 (issues found), or 2 (error).
            # We treat both 0 and 1 as valid JSON output; only 2 is a real error.
            if result.returncode == 2:
                print(f"    Error running bandit: {result.stderr.strip()}")
                continue

            # --- Parse JSON results ---
            try:
                report = json.loads(result.stdout)
            except json.JSONDecodeError:
                print(f"    Could not parse bandit output: {result.stdout[:200]}")
                continue

            issues = report.get("results", [])

            if not issues:
                print("    No issues found.\n")
            else:
                print(f"    Found {len(issues)} issue(s):\n")
                for issue in issues:
                    print(f"      Line {issue.get('line_number', '?'):>4}  "
                          f"[{issue.get('issue_severity', '?'):<6} / "
                          f"{issue.get('issue_confidence', '?'):<6}]  "
                          f"{issue.get('issue_text', '')}")
                print()

            bandit_results.append({"filename": filename, "issues": issues})

        finally:
            # --- Always clean up the temp file ---
            os.unlink(tmp.name)

    return bandit_results



def run_security_agent(bandit_results, python_diffs):
    """
    Security Agent: combines bandit's raw findings with the actual code diff
    and asks the LLM to produce a structured security verdict.

    Uses ChatGroq.with_structured_output() so the LLM is forced to respond
    in the exact shape of SecurityAgentOutput — no free-form text parsing needed.

    Args:
        bandit_results: list of dicts returned by run_bandit_scan()
                        [{"filename": ..., "issues": [bandit finding dicts]}, ...]
        python_diffs:   list of dicts built in main()
                        [{"filename", "status", "additions", "deletions", "patch"}, ...]

    Returns:
        SecurityAgentOutput Pydantic object
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment. Check your .env file.")
        sys.exit(1)

    print(f"\n{'='*50}")
    print("  SECURITY AGENT (LLM JUDGMENT)")
    print(f"{'='*50}\n")

    # --- Build the prompt ---
    # Section 1: bandit findings (all files combined, with severity/confidence labels)
    bandit_section_lines = []
    total_issues = 0
    for entry in bandit_results:
        fname = entry["filename"]
        issues = entry["issues"]
        bandit_section_lines.append(f"File: {fname}")
        if not issues:
            bandit_section_lines.append("  bandit: no issues found")
        else:
            for iss in issues:
                total_issues += 1
                bandit_section_lines.append(
                    f"  Line {iss.get('line_number', '?')}: "
                    f"[{iss.get('issue_severity', '?')} / {iss.get('issue_confidence', '?')}] "
                    f"{iss.get('issue_text', '')} "
                    f"(test_id: {iss.get('test_id', '?')})"
                )
        bandit_section_lines.append("")
    bandit_section = "\n".join(bandit_section_lines)

    # Section 2: actual code diff (full patch) for each file — gives the LLM
    # the real code context so it can judge whether a finding is a real risk
    # or bandit flagging something benign (e.g., a test mock, an example, etc.)
    diff_section_lines = []
    patch_lookup = {d["filename"]: d["patch"] for d in python_diffs}
    for entry in bandit_results:
        fname = entry["filename"]
        patch = patch_lookup.get(fname, "(patch not available)")
        diff_section_lines.append(f"=== Diff for {fname} ===")
        diff_section_lines.append(patch)
        diff_section_lines.append("")
    diff_section = "\n".join(diff_section_lines)

    user_prompt = f"""You are a senior security-focused code reviewer. Below are:
1. Static analysis findings from bandit for each changed Python file in this pull request.
2. The actual code diff (patch) for each file, so you can judge each finding in context.

Your job:
- Identify which bandit findings represent GENUINE security concerns given the actual code.
- Identify which are likely false positives (e.g., flagged in test fixtures, demo/example code,
  or patterns that are safe in this specific usage context).
- Assign an overall severity across all real findings: one of none / low / medium / high / critical.
- Write 2-4 sentences of reasoning explaining your overall judgment.

--- BANDIT FINDINGS ({total_issues} total) ---
{bandit_section}
--- CODE DIFFS ---
{diff_section}
"""

    # --- Create the structured LLM client ---
    # with_structured_output() wraps the LLM so its response is parsed directly
    # into a SecurityAgentOutput Pydantic object — the LLM cannot return
    # free-form text; it must conform to the schema.
    base_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
    )
    structured_llm = base_llm.with_structured_output(SecurityAgentOutput)

    system_msg = SystemMessage(
        content="You are an expert Python security reviewer. Always respond with structured "
                "JSON conforming exactly to the provided schema."
    )

    print(f"  Sending {total_issues} bandit finding(s) + diff context to LLM...")
    result: SecurityAgentOutput = structured_llm.invoke(
        [system_msg, HumanMessage(content=user_prompt)]
    )

    # --- Pretty-print the structured result ---
    print(f"\n  Overall Severity : {result.overall_severity.upper()}")
    print(f"  Reasoning        : {result.reasoning}")

    print(f"\n  Real Issues ({len(result.real_issues)}):")
    if result.real_issues:
        for issue in result.real_issues:
            print(f"    [{issue.file}:{issue.line_number}] {issue.description}")
            print(f"      Why it matters: {issue.why_it_matters}")
    else:
        print("    (none)")

    print(f"\n  Dismissed as Noise ({len(result.dismissed_noise)}):")
    if result.dismissed_noise:
        for note in result.dismissed_noise:
            print(f"    - {note}")
    else:
        print("    (none)")

    print(f"\n{'='*50}\n")
    return result


def main():
    args = parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN not found in environment. Check your .env file.")
        sys.exit(1)

    g = Github(auth=Auth.Token(token))

    # Fetch repo
    try:
        repo = g.get_repo(args.repo)
    except GithubException as e:
        if e.status == 404:
            print(f"Error: Repository '{args.repo}' not found. Check the owner/reponame format.")
        else:
            print(f"Error fetching repository: {e.data.get('message', str(e))}")
        sys.exit(1)

    # Fetch PR
    try:
        pr = repo.get_pull(args.pr)
    except GithubException as e:
        if e.status == 404:
            print(f"Error: PR #{args.pr} not found in '{args.repo}'.")
        else:
            print(f"Error fetching PR: {e.data.get('message', str(e))}")
        sys.exit(1)

    # Print PR metadata
    print(f"\n{'='*50}")
    print(f"  PR #{pr.number}: {pr.title}")
    print(f"{'='*50}")
    print(f"  Author       : {pr.user.login}")
    print(f"  State        : {pr.state}")
    print(f"  Files changed: {pr.changed_files}")
    print(f"  Additions    : +{pr.additions}")
    print(f"  Deletions    : -{pr.deletions}")
    print(f"{'='*50}\n")

    # Fetch changed files and filter to Python-only
    print("Fetching changed files...\n")
    files = list(pr.get_files())

    python_files = [f for f in files if f.filename.endswith(".py")]
    skipped_count = len(files) - len(python_files)

    print(f"  Total files in PR : {len(files)}")
    print(f"  Python files found: {len(python_files)}")
    print(f"  Non-Python files skipped: {skipped_count}\n")

    # In-memory storage for full diff data (passed to agents later)
    python_diffs = []

    # Print details for each Python file
    print(f"{'='*50}")
    print("  CHANGED PYTHON FILES")
    print(f"{'='*50}\n")

    for f in python_files:
        print(f"  File      : {f.filename}")
        print(f"  Status    : {f.status}")
        print(f"  Additions : +{f.additions}")
        print(f"  Deletions : -{f.deletions}")

        patch = f.patch or ""
        truncated_patch = patch[:500] + ("..." if len(patch) > 500 else "")
        print(f"  Patch (truncated to 500 chars):\n{truncated_patch}")
        print(f"\n{'-'*50}\n")

        # Store full (non-truncated) diff data
        python_diffs.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "patch": patch,
        })

    print(f"[INFO] Stored full diff data for {len(python_diffs)} Python file(s) in memory.")

    # Step 3: Test LLM connection (isolated, no diff logic yet)
    test_llm_connection()

    # Step 4: Static security analysis with bandit
    bandit_results = run_bandit_scan(python_files, repo, pr)

    # Step 5: Security Agent — LLM judges bandit findings against the diff
    if bandit_results:
        run_security_agent(bandit_results, python_diffs)
    else:
        print("\n[INFO] No bandit results to analyse — skipping Security Agent.")


if __name__ == "__main__":
    main()
