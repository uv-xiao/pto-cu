# Setup

Initialize GitHub workflow state.

## Authenticate

Run:

```bash
gh auth status
```

If authentication fails, tell the user to run `gh auth login` and stop.

## Detect Canonical Repository

Parse `origin` first. For `pto-cu`, `origin` is the canonical PR repository:
expected target `uv-xiao/pto-cu`, base `main`, and base ref `origin/main`.
Use `gh repo view` only as a fallback or consistency check.

```bash
ORIGIN_URL=$(git remote get-url origin 2>/dev/null || echo "")
REPO_PATH=$(echo "$ORIGIN_URL" | sed -n 's#.*[:/]\([^/]*\/[^/]*\)$#\1#p')
REPO_OWNER=${REPO_PATH%%/*}
REPO_NAME=${REPO_PATH#*/}
REPO_NAME=${REPO_NAME%.git}

GH_REPO_OWNER=$(gh repo view --json owner -q '.owner.login' 2>/dev/null || echo "")
GH_REPO_NAME=$(gh repo view --json name -q '.name' 2>/dev/null || echo "")
GH_DEFAULT_BRANCH=$(gh repo view --json defaultBranchRef -q '.defaultBranchRef.name' 2>/dev/null || echo "")

if [ "$REPO_OWNER/$REPO_NAME" = "uv-xiao/pto-cu" ]; then
  PR_REPO_OWNER="uv-xiao"
  PR_REPO_NAME="pto-cu"
  DEFAULT_BRANCH="main"
  if [ -n "$GH_REPO_OWNER$GH_REPO_NAME" ] \
    && [ "$GH_REPO_OWNER/$GH_REPO_NAME" != "uv-xiao/pto-cu" ]; then
    echo "Ignoring stale gh repo view context: $GH_REPO_OWNER/$GH_REPO_NAME"
  fi
elif [ -n "$GH_REPO_OWNER" ] && [ -n "$GH_REPO_NAME" ]; then
  PR_REPO_OWNER="$GH_REPO_OWNER"
  PR_REPO_NAME="$GH_REPO_NAME"
  DEFAULT_BRANCH="${GH_DEFAULT_BRANCH:-main}"
else
  PR_REPO_OWNER="uv-xiao"
  PR_REPO_NAME="pto-cu"
  DEFAULT_BRANCH="main"
fi
```

If `origin` is `git@github.com:uv-xiao/pto-cu.git` but `gh repo view`
returns another repository such as `hw-native-sys/simpler`, treat
`gh repo view` as stale context. Do not add an `upstream` remote from that
stale value.

## Detect Role And Remotes

```bash
if [ "$REPO_OWNER" = "$PR_REPO_OWNER" ] && [ "$REPO_NAME" = "$PR_REPO_NAME" ]; then
  ROLE="owner"
  BASE_REMOTE="origin"
  PR_HEAD_PREFIX=""
else
  ROLE="fork"
  BASE_REMOTE="upstream"
  PR_HEAD_PREFIX="$REPO_OWNER:"
  if ! git remote | grep -q '^upstream$'; then
    git remote add upstream "git@github.com:$PR_REPO_OWNER/$PR_REPO_NAME.git"
  fi
fi

git fetch "$BASE_REMOTE" "$DEFAULT_BRANCH"
git fetch origin

BASE_REF="$BASE_REMOTE/$DEFAULT_BRANCH"
PUSH_REMOTE="origin"
BRANCH_NAME=$(git branch --show-current 2>/dev/null || echo "")
UNCOMMITTED=$(git status --porcelain)
COMMITS_AHEAD=$(git rev-list HEAD --not "$BASE_REF" --count 2>/dev/null || echo "0")
```

Never assume local `main` is fresh. Use `BASE_REF`.
