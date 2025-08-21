"""
Simple Git helper: stage all changes, ask for a commit message, commit, and push.

Usage:
  python push_to_github.py

Requirements:
  - git installed and available in PATH
  - this script is executed inside a git repository with a configured remote
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def run(cmd: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def ensure_git_available():
    if shutil.which("git") is None:
        print("Error: git is not installed or not on PATH.")
        sys.exit(1)


def ensure_git_repo():
    r = run(["git", "rev-parse", "--is-inside-work-tree"])
    if r.returncode != 0 or r.stdout.strip() != "true":
        print("Error: Not inside a git repository.")
        sys.exit(1)


def current_branch() -> str:
    r = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if r.returncode != 0:
        print(r.stderr.strip() or "Error: Unable to determine current branch.")
        sys.exit(1)
    return r.stdout.strip()


def has_upstream() -> bool:
    r = run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    return r.returncode == 0 and bool(r.stdout.strip())


def origin_exists() -> bool:
    r = run(["git", "remote"])
    if r.returncode != 0:
        return False
    return "origin" in {x.strip() for x in r.stdout.splitlines()}


def working_changes_exist() -> bool:
    # staged or unstaged changes
    r = run(["git", "status", "--porcelain"])
    return r.returncode == 0 and bool(r.stdout.strip())


def main():
    ensure_git_available()
    ensure_git_repo()

    # Stage everything (new, modified, deleted)
    run(["git", "add", "-A"])  # don't hard fail on add; commit will reflect actual changes

    # Ask for commit message
    try:
        commit_msg = input("Enter commit message: ").strip()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)

    while not commit_msg:
        try:
            commit_msg = input("Commit message cannot be empty. Enter commit message: ").strip()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(130)

    # If no changes, inform and still try pushing latest
    if not working_changes_exist():
        print("No local changes detected to commit. Proceeding to push.")
    else:
        c = run(["git", "commit", "-m", commit_msg])
        if c.returncode != 0:
            # If nothing to commit, continue to push; otherwise, show error
            msg = (c.stderr or c.stdout).strip()
            if "nothing to commit" in msg.lower():
                print("Nothing to commit. Proceeding to push.")
            else:
                print(msg or "Commit failed.")
                sys.exit(c.returncode)
        else:
            print("Commit created.")

    branch = current_branch()

    if has_upstream():
        p = run(["git", "push"])
    else:
        if not origin_exists():
            print("Error: No 'origin' remote configured. Add a remote and re-run:")
            print("  git remote add origin <YOUR_REPO_URL>")
            sys.exit(1)
        p = run(["git", "push", "-u", "origin", branch])

    if p.returncode != 0:
        print((p.stderr or p.stdout).strip() or "Push failed.")
        # Common hint for auth issues
        print("Hint: Ensure you are authenticated (SSH keys or HTTPS with a token).")
        sys.exit(p.returncode)

    print(f"Pushed to origin/{branch}.")


if __name__ == "__main__":
    main()
