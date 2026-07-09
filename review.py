import argparse
import sys

from dotenv import load_dotenv
import os
from github import Github, Auth, GithubException

# Load environment variables from .env
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="PR Review Agent — fetch PR metadata")
    parser.add_argument("--repo", required=True, help="Repository in 'owner/reponame' format")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    return parser.parse_args()


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


if __name__ == "__main__":
    main()
