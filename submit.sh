#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./submit.sh
#   ./submit.sh "Finish experiment analysis"

COMMIT_MESSAGE="${1:-Update research files}"

# Make sure we're inside a Git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: This is not a Git repository."
    exit 1
fi

echo "Adding Markdown, Python, and Jupyter Notebook files..."

# Find relevant files and add them safely, including filenames with spaces.
find . \
    -type f \
    \( -name "*.md" -o -name "*.py" -o -name "*.ipynb" \) \
    -not -path "./.git/*" \
    -print0 |
    xargs -0 -r git add --

# Check whether anything was staged
if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

echo "Committing..."
git commit -m "$COMMIT_MESSAGE"

BRANCH="$(git branch --show-current)"

if [[ -z "$BRANCH" ]]; then
    echo "Error: Could not determine current Git branch."
    exit 1
fi

echo "Pushing branch: $BRANCH"

# Push and set upstream automatically if necessary
git push -u origin "$BRANCH"

echo "Done."