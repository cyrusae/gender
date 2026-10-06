#!/usr/bin/env bash
# Make /tmp/gender.tgz from the committed files, recording the commit so results produced on
# the pod (which has no git history) can say which code made them. Refuses if code is dirty.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain --untracked-files=no -- . ':!results')" ]; then
  echo "Uncommitted changes outside results/: commit first, so the bundle matches a commit." >&2
  exit 1
fi
git rev-parse HEAD > BUNDLE_COMMIT.txt
git ls-files > /tmp/files.txt
echo BUNDLE_COMMIT.txt >> /tmp/files.txt
# --no-xattrs / COPYFILE_DISABLE: no macOS metadata (it triggers warnings on Linux)
COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -czf /tmp/gender.tgz -T /tmp/files.txt
rm BUNDLE_COMMIT.txt
echo "bundle: /tmp/gender.tgz at $(git rev-parse --short HEAD) ($(du -h /tmp/gender.tgz | cut -f1))"
