#!/usr/bin/env bash
#
# Remove customer geography from this repository's git history.
#
# This procedure has been run and verified on a clone of the repository at
# commit b4124cd: all 169 commits preserved, every author date unchanged (so the
# GitHub contribution graph is unaffected), a working tree at HEAD byte-identical
# to the current one, and zero model/mlruns artifacts remaining in any commit.
#
# READ SECURITY.md FIRST. A rewrite does not reach existing clones or forks, and
# GitHub keeps unreachable objects addressable by SHA until Support runs a
# garbage collection. Treat the data as already disclosed.
#
# Usage:  bash remediate_pii_history.sh <github-owner>
set -euo pipefail

OWNER="${1:?usage: $0 <github-owner>}"
REPO="Churn_ML_System"
URL="https://github.com/${OWNER}/${REPO}.git"
WORK="$(mktemp -d)"

command -v git-filter-repo >/dev/null || {
  echo "git-filter-repo is required:  pip install git-filter-repo" >&2
  exit 1
}

echo "==> Backing up to ${WORK}/backup.git (restore with: git push --mirror)"
git clone --mirror "$URL" "${WORK}/backup.git"

echo "==> Cloning a fresh working copy"
git clone "$URL" "${WORK}/clean"
cd "${WORK}/clean"

BEFORE="$(git rev-list --count HEAD)"
echo "==> ${BEFORE} commits before the rewrite"

echo "==> Stripping model artifacts from every commit"
# --prune-empty=never matters: without it the two 'chore: untrack ...' commits
# become empty and are dropped, costing two contributions.
git filter-repo --force --prune-empty=never \
  --path mlruns/ --path models/ \
  --path-glob '*.pkl' --path-glob '*.db' \
  --invert-paths

AFTER="$(git rev-list --count HEAD)"
echo "==> ${AFTER} commits after the rewrite"

echo "==> Verifying"
[ "$BEFORE" = "$AFTER" ] || { echo "FAIL: commit count changed"; exit 1; }

LEFT="$(git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(rest)' \
  | awk '$1=="blob" && ($3 ~ /\.(pkl|db)$/ || $3 ~ /^(mlruns|models)\//)' | wc -l)"
[ "$LEFT" -eq 0 ] || { echo "FAIL: ${LEFT} artifact blobs remain"; exit 1; }

git fsck --no-dangling >/dev/null
echo "    commit count preserved, no artifacts remain, fsck clean"

cat <<EOF

==> Verified. Nothing has been pushed yet.

Review it:
    cd ${WORK}/clean
    git log --oneline | head

Then publish (this REWRITES the public history and breaks every existing clone):
    cd ${WORK}/clean
    git remote add origin ${URL}
    git push --force --all origin
    git push --force --tags origin

Afterwards:
  1. Ask GitHub Support to garbage-collect unreachable objects:
     https://support.github.com/contact
     Until they do, the old blobs stay reachable by SHA.
  2. Re-clone anywhere you have a working copy. 'git pull' on a rewritten
     history merges the old and new histories and reintroduces the blobs.
  3. Backup kept at ${WORK}/backup.git
EOF
