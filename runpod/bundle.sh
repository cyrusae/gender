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
# Gitignored inputs the pod needs at run time (steering directions use the Phase 2/3 covariates:
# lexicon frequencies and concreteness norms): fail loudly if missing rather than on the pod.
for f in data/lexicon/es_nouns.csv data/lexicon/de_nouns.csv \
         data/raw/norms/Concreteness_ratings_Brysbaert_et_al_BRM.txt data/raw/nltk_data; do
  [ -e "$f" ] || { echo "missing runtime input $f" >&2; exit 1; }
  echo "$f" >> /tmp/files.txt
done
# --no-xattrs / COPYFILE_DISABLE: no macOS metadata (it triggers warnings on Linux)
COPYFILE_DISABLE=1 tar -h --no-mac-metadata --no-xattrs -czf /tmp/gender.tgz -T /tmp/files.txt
rm BUNDLE_COMMIT.txt
echo "bundle: /tmp/gender.tgz at $(git rev-parse --short HEAD) ($(du -h /tmp/gender.tgz | cut -f1))"
