import argparse
import json
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import os
from github import Github, Auth, GithubException
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

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


class CoverageGap(BaseModel):
    """A single coverage gap in the changed files, with risk assessment."""
    file: str = Field(description="Filename with the coverage gap")
    uncovered_lines: List[int] = Field(description="List of line numbers not covered by tests")
    risk_note: str = Field(
        description="Brief note on the risk of these specific lines being untested"
    )

class CoverageAgentOutput(BaseModel):
    """Structured output from the Coverage Agent LLM call."""
    overall_risk: str = Field(
        description="Overall risk from missing coverage. Must be: 'none', 'low', 'medium', 'high'."
    )
    files_with_gaps: List[CoverageGap] = Field(
        description="List of files with missing coverage and risk assessments"
    )
    reasoning: str = Field(
        description="2-4 sentences explaining why the untested lines matter (or don't)"
    )

class RiskVerdict(BaseModel):
    """Final aggregated verdict from the Risk Aggregator LLM call."""
    risk_score: int = Field(description="A score from 0 to 100 representing overall PR risk")
    recommendation: str = Field(
        description="Final decision: 'ready', 'needs_changes', or 'block'"
    )
    blocking_issues: List[str] = Field(
        description="List of specific issues that must be addressed before merging, if any"
    )
    summary: str = Field(
        description="3-5 sentences written as if explaining to a human reviewer"
    )


# ---------------------------------------------------------------------------
# LangGraph shared state schema
# ---------------------------------------------------------------------------

class PRReviewState(TypedDict, total=False):
    """
    Shared state object passed between every node in the LangGraph.

    Each node receives the full state, does its work, and returns a dict
    with only the fields it wants to update.  LangGraph merges those updates
    back into the state before passing it to the next node.

    Fields
    ------
    repo_name             : "owner/reponame" string from CLI
    pr_number             : PR number integer from CLI
    repo                  : PyGithub Repository object (set in main, passed via state)
    pr                    : PyGithub PullRequest object
    python_files          : raw PullRequestFile list from pr.get_files()
    diff_files            : list of dicts {filename, status, additions, deletions, patch}
    bandit_results        : output of run_bandit_scan()
    security_agent_output : SecurityAgentOutput Pydantic object from run_security_agent()
    coverage_results      : dict with 'has_tests' and 'coverage_data'
    coverage_agent_output : CoverageAgentOutput Pydantic object
    risk_verdict          : RiskVerdict Pydantic object
    """
    repo_name: str
    pr_number: int
    repo: Any                           # PyGithub Repository — not JSON-serialisable, kept in memory
    pr: Any                             # PyGithub PullRequest
    python_files: List[Any]             # PullRequestFile objects
    diff_files: List[Dict[str, Any]]    # [{filename, status, additions, deletions, patch}, ...]
    bandit_results: List[Dict[str, Any]]
    security_agent_output: Optional[SecurityAgentOutput]
    coverage_results: Dict[str, Any]
    coverage_agent_output: Optional[CoverageAgentOutput]
    risk_verdict: Optional[RiskVerdict]


def parse_args():
    parser = argparse.ArgumentParser(description="PR Review Agent — fetch PR metadata")
    parser.add_argument("--repo", required=True, help="Repository in 'owner/reponame' format")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    return parser.parse_args()


def test_llm_connection():
    """Verify GROQ_API_KEY is present and the ChatGroq client can be instantiated."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment. Check your .env file.")
        sys.exit(1)
    # Instantiate the client — if the key is malformed this raises immediately.
    ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
    print("[OK] Groq LLM client initialised successfully.")


def run_security_agent(bandit_results, python_diffs):
    """
    Security Agent: takes raw bandit findings and the PR diffs,
    and asks the LLM to filter for genuine security issues.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment.")
        sys.exit(1)

    print(f"\n{'='*50}")
    print("  SECURITY AGENT (LLM JUDGMENT)")
    print(f"{'='*50}\n")

    # Construct the context for the LLM
    context = "Files being reviewed:\n"
    for diff in python_diffs:
        context += f"Filename: {diff['filename']}\nPatch snippet:\n{diff['patch']}\n\n"

    prompt = f"""You are a senior security engineer. Analyze the following Bandit security tool findings within the context of the provided code diffs.
Identify which findings are real security vulnerabilities, which are false positives, and which are noise (like test files).

--- CODE DIFFS ---
{context}

--- BANDIT FINDINGS ---
{json.dumps(bandit_results, indent=2)}
"""

    base_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
    structured_llm = base_llm.with_structured_output(SecurityAgentOutput)

    system_msg = SystemMessage(
        content="You are an expert security reviewer. Respond only with structured JSON."
    )

    result: SecurityAgentOutput = structured_llm.invoke(
        [system_msg, HumanMessage(content=prompt)]
    )

    print(f"  Overall Severity: {result.overall_severity.upper()}")
    print(f"  Issues Found    : {len(result.real_issues)}")
    print(f"{'='*50}\n")
    return result


def run_coverage_agent(coverage_results, python_diffs):
    """
    Coverage Agent: combines missing coverage lines with actual code diff
    and asks the LLM to judge the risk of those untested lines.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment.")
        sys.exit(1)

    print(f"\n{'='*50}")
    print("  COVERAGE AGENT (LLM JUDGMENT)")
    print(f"{'='*50}\n")

    if not coverage_results.get("has_tests"):
        print("  [INFO] No tests or no coverage data found. Generating fallback report...")
        # If there are no tests, we still want the agent to output a result
        base_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
        structured_llm = base_llm.with_structured_output(CoverageAgentOutput)
        prompt = (
            "The repository has no tests or tests failed to run. "
            "Please assess the overall risk of merging untested code based on these diffs:\n\n"
        )
        for d in python_diffs:
            prompt += f"=== {d['filename']} ===\n{d['patch']}\n\n"
        
        result = structured_llm.invoke([
            SystemMessage(content="You are an expert QA and code reviewer."),
            HumanMessage(content=prompt)
        ])
        return result

    cov_data = coverage_results.get("coverage_data", [])
    if not cov_data:
        # All changed files are covered
        print("  [INFO] 100% coverage on changed files! Returning zero-risk output.")
        return CoverageAgentOutput(
            overall_risk="none",
            files_with_gaps=[],
            reasoning="All modified lines in the examined files are fully covered by tests. Great job!"
        )

    # Build prompt for files with missing coverage
    cov_section = ""
    total_gaps = 0
    for item in cov_data:
        fname = item["file"]
        missing = item["missing_lines"]
        total_gaps += len(missing)
        cov_section += f"File: {fname}\n  Uncovered lines: {missing}\n\n"

    diff_section_lines = []
    patch_lookup = {d["filename"]: d["patch"] for d in python_diffs}
    for item in cov_data:
        fname = item["file"]
        patch = patch_lookup.get(fname, "(patch not available)")
        diff_section_lines.append(f"=== Diff for {fname} ===")
        diff_section_lines.append(patch)
        diff_section_lines.append("")
    diff_section = "\n".join(diff_section_lines)

    user_prompt = f"""You are a senior QA engineer and code reviewer. Below are:
1. Coverage report showing which lines in the modified files are NOT covered by tests.
2. The actual code diff (patch) for each file.

Your job:
- Identify the logic on the uncovered lines.
- Assess how risky it is that these specific lines are untested (e.g. is it core business logic, or just a simple print statement?).
- Assign an overall risk for the missing coverage: none / low / medium / high.
- Provide reasoning.

--- UNCOVERED LINES ---
{cov_section}
--- CODE DIFFS ---
{diff_section}
"""

    base_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
    structured_llm = base_llm.with_structured_output(CoverageAgentOutput)

    system_msg = SystemMessage(
        content="You are an expert Python QA reviewer. Always respond with structured "
                "JSON conforming exactly to the provided schema."
    )

    print(f"  Sending {total_gaps} uncovered line(s) + diff context to LLM...")
    result: CoverageAgentOutput = structured_llm.invoke(
        [system_msg, HumanMessage(content=user_prompt)]
    )

    print(f"\n  Overall Risk : {result.overall_risk.upper()}")
    print(f"  Reasoning    : {result.reasoning}")
    if result.files_with_gaps:
        for gap in result.files_with_gaps:
            print(f"    - {gap.file}: {gap.uncovered_lines}")
            print(f"      Risk: {gap.risk_note}")
    print(f"\n{'='*50}\n")
    return result


def security_agent_node(state: PRReviewState) -> PRReviewState:
    """
    LangGraph node — Security Agent.
    """
    print("\n[GRAPH] Entering node: security_agent_node")

    bandit_results = run_bandit_scan(
        state["python_files"],
        state["repo"],
        state["pr"],
    )

    if bandit_results:
        security_output = run_security_agent(bandit_results, state["diff_files"])
    else:
        print("\n[INFO] No bandit results to analyse — skipping Security Agent.")
        security_output = None

    return {
        "bandit_results": bandit_results,
        "security_agent_output": security_output,
    }


def coverage_agent_node(state: PRReviewState) -> PRReviewState:
    """
    LangGraph node — Coverage Agent.
    """
    print("\n[GRAPH] Entering node: coverage_agent_node")

    coverage_results = run_coverage_scan(
        state["python_files"],
        state["repo"],
        state["pr"],
    )

    coverage_output = run_coverage_agent(coverage_results, state["diff_files"])

    return {
        "coverage_results": coverage_results,
        "coverage_agent_output": coverage_output,
    }


def run_risk_aggregator(security_output, coverage_output):
    """
    Risk Aggregator: synthesizes the findings of the Security and Coverage agents
    into a final RiskVerdict.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment.")
        sys.exit(1)

    print(f"\n{'='*50}")
    print("  RISK AGGREGATOR (SYNTHESIS)")
    print(f"{'='*50}\n")

    # Serialize agent outputs to pass to the LLM
    sec_json = security_output.model_dump_json() if security_output else "{}"
    cov_json = coverage_output.model_dump_json() if coverage_output else "{}"

    prompt = f"""You are the Lead Code Reviewer. You have received reports from two specialized agents:
1. Security Agent
2. Test Coverage Agent

Your job is to synthesize these reports into a final `RiskVerdict` for the pull request.
Consider:
- High severity security issues should likely 'block' the PR.
- Critical logic with no coverage might 'need_changes' or 'block'.
- If both are clear, the PR is 'ready'.

--- SECURITY AGENT OUTPUT ---
{sec_json}

--- COVERAGE AGENT OUTPUT ---
{cov_json}
"""

    base_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
    structured_llm = base_llm.with_structured_output(RiskVerdict)

    system_msg = SystemMessage(
        content="You are a Lead Code Reviewer. Always respond with structured JSON."
    )

    print("  Synthesizing final risk verdict...")
    result: RiskVerdict = structured_llm.invoke([system_msg, HumanMessage(content=prompt)])
    
    print(f"  Final Recommendation: {result.recommendation.upper()}")
    print(f"{'='*50}\n")
    return result


def risk_aggregator_node(state: PRReviewState) -> PRReviewState:
    """
    LangGraph node — Risk Aggregator.
    Runs after both security and coverage nodes have completed.
    """
    print("\n[GRAPH] Entering node: risk_aggregator_node")
    
    risk_verdict = run_risk_aggregator(
        state.get("security_agent_output"),
        state.get("coverage_agent_output")
    )
    
    return {
        "risk_verdict": risk_verdict
    }


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Uses a fan-out/fan-in pattern:
    START ─┬─► security_agent_node ─┬─► risk_aggregator_node ─► END
           └─► coverage_agent_node ─┘
    """
    graph = StateGraph(PRReviewState)

    # Register nodes
    graph.add_node("security_agent_node", security_agent_node)
    graph.add_node("coverage_agent_node", coverage_agent_node)
    graph.add_node("risk_aggregator_node", risk_aggregator_node)

    # Parallel fan-out from START
    graph.add_edge(START, "security_agent_node")
    graph.add_edge(START, "coverage_agent_node")
    
    # Fan-in to the aggregator
    graph.add_edge("security_agent_node", "risk_aggregator_node")
    graph.add_edge("coverage_agent_node", "risk_aggregator_node")
    
    # Execution path finishes after aggregator
    graph.add_edge("risk_aggregator_node", END)

    return graph.compile()


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


def run_coverage_scan(python_files, repo, pr):
    """
    Clones the PR head repo into a temp directory and runs coverage.
    Returns:
        dict: {"has_tests": bool, "coverage_data": [{"file": str, "missing_lines": list[int]}, ...]}
    """
    print(f"\n{'='*50}")
    print("  TEST COVERAGE SCAN")
    print(f"{'='*50}\n")

    target_files = [f.filename for f in python_files if f.status != "removed"]
    if not target_files:
        print("  No Python files to check coverage for.\n")
        return {"has_tests": False, "coverage_data": []}

    clone_url = pr.head.repo.clone_url
    head_ref = pr.head.ref

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  Cloning {clone_url} branch {head_ref}...")
        try:
            # Clone only the specific branch, depth 1
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", head_ref, clone_url, "."],
                cwd=tmpdir, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"  Warning: failed to clone repo: {e.stderr}")
            return {"has_tests": False, "coverage_data": []}

        print("  Running tests with coverage...")
        # Fallback to unittest discover if pytest is missing, 
        # but coverage module is assumed present.
        subprocess.run(
            [sys.executable, "-m", "coverage", "run", "-m", "unittest", "discover"],
            cwd=tmpdir, capture_output=True, text=True
        )
        subprocess.run(
            [sys.executable, "-m", "coverage", "json"],
            cwd=tmpdir, capture_output=True, text=True
        )

        cov_file = os.path.join(tmpdir, "coverage.json")
        if not os.path.exists(cov_file):
            print("  Warning: coverage.json not generated. Likely no tests found.")
            return {"has_tests": False, "coverage_data": []}

        print("  Parsing coverage results...")
        try:
            with open(cov_file, "r") as f:
                cov_report = json.load(f)
        except Exception as e:
            print(f"  Warning: failed to read coverage.json: {e}")
            return {"has_tests": False, "coverage_data": []}

        coverage_data = []
        cov_files = cov_report.get("files", {})

        for fname in target_files:
            # Map github filename to coverage.json keys
            file_cov = cov_files.get(fname)
            if not file_cov:
                for k, v in cov_files.items():
                    if k.endswith(fname):
                        file_cov = v
                        break

            if file_cov:
                missing = file_cov.get("missing_lines", [])
                if missing:
                    coverage_data.append({"file": fname, "missing_lines": missing})

        if not coverage_data:
            print("  No missing coverage found for the changed files!\n")
        else:
            print(f"  Found missing coverage in {len(coverage_data)} file(s).\n")

        return {"has_tests": True, "coverage_data": coverage_data}


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

    # Step 1: Verify LLM client (quick, no API call)
    test_llm_connection()

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

    # Step 2: Print PR metadata
    print(f"\n{'='*50}")
    print(f"  PR #{pr.number}: {pr.title}")
    print(f"{'='*50}")
    print(f"  Author       : {pr.user.login}")
    print(f"  State        : {pr.state}")
    print(f"  Files changed: {pr.changed_files}")
    print(f"  Additions    : +{pr.additions}")
    print(f"  Deletions    : -{pr.deletions}")
    print(f"{'='*50}\n")

    # Step 3: Fetch changed files and build diff list (Python-only)
    print("Fetching changed files...\n")
    files = list(pr.get_files())

    python_files = [f for f in files if f.filename.endswith(".py")]
    skipped_count = len(files) - len(python_files)

    print(f"  Total files in PR : {len(files)}")
    print(f"  Python files found: {len(python_files)}")
    print(f"  Non-Python files skipped: {skipped_count}\n")

    diff_files = []

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

        diff_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "patch": patch,
        })

    print(f"[INFO] Stored full diff data for {len(diff_files)} Python file(s) in memory.")

    # Step 4 + 5: Run the LangGraph — bandit scan + security agent inside the graph
    print("\n[GRAPH] Building and invoking the PR Review StateGraph...")
    graph = build_graph()

    initial_state: PRReviewState = {
        "repo_name": args.repo,
        "pr_number": args.pr,
        "repo": repo,
        "pr": pr,
        "python_files": python_files,
        "diff_files": diff_files,
        "bandit_results": [],
        "security_agent_output": None,
        "coverage_results": {},
        "coverage_agent_output": None,
        "risk_verdict": None,
    }

    final_state: PRReviewState = graph.invoke(initial_state)

    # Print final structured output from state
    risk: Optional[RiskVerdict] = final_state.get("risk_verdict")
    if risk:
        print(f"\n{'='*50}")
        print("  FINAL AGGREGATED RISK VERDICT (from graph state)")
        print(f"{'='*50}")
        print(f"  Recommendation : {risk.recommendation.upper()}")
        print(f"  Risk Score     : {risk.risk_score} / 100")
        print(f"  Summary        : {risk.summary}")
        
        if risk.blocking_issues:
            print(f"\n  Blocking Issues ({len(risk.blocking_issues)}):")
            for issue in risk.blocking_issues:
                print(f"    - {issue}")
        else:
            print("\n  Blocking Issues: None")
        print(f"{'='*50}\n")
    else:
        print("\n[INFO] No risk verdict in final graph state.")


if __name__ == "__main__":
    main()
