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


if __name__ == "__main__":
    main()
