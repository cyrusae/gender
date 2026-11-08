#!/usr/bin/env bash
# Download the pretrained embeddings both upstream repos need, onto the store (not the repo),
# then link them where each notebook expects them.
#   Zhou et al. 2019: MUSE aligned fastText Wikipedia vectors wiki.multi.{en,es,fr,de}.vec
#                     + MUSE bilingual dictionaries (en-es, es-en, es-fr, en-fr, fr-en)
#   Omrani Sabbaghi & Caliskan 2022: fastText Common Crawl cc.{es,fr,de,it,pl,en}.300.vec
# Resumable (curl -C -). Sizes: MUSE ~0.63 GB each; cc ~1.3 GB gzipped, ~4.5 GB unzipped each.
set -euo pipefail
STORE=${GBLEED_STORE:-/Volumes/Lexar/gbleed}/shared/embeddings
EXT=$(cd "$(dirname "$0")/../external" && pwd)
mkdir -p "$STORE/muse/dictionaries" "$STORE/fasttext_cc"
get() { curl -sSfL --retry 5 -C - -o "$2" "$1"; echo "got $(basename "$2") $(du -h "$2" | cut -f1)"; }

for l in en es fr de; do
  get "https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.$l.vec" "$STORE/muse/wiki.multi.$l.vec"
done
for d in en-es es-en es-fr en-fr fr-en; do
  get "https://dl.fbaipublicfiles.com/arrival/dictionaries/$d.txt" "$STORE/muse/dictionaries/$d.txt"
done
for l in es fr de it pl en; do
  f="$STORE/fasttext_cc/cc.$l.300.vec"
  [ -s "$f" ] && { echo "have $(basename "$f")"; continue; }
  get "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.$l.300.vec.gz" "$f.gz"
  gunzip -c "$f.gz" > "$f.part" && mv "$f.part" "$f" && rm "$f.gz"
  echo "unzipped $(basename "$f") $(du -h "$f" | cut -f1)"
done
shasum -a 256 "$STORE"/muse/*.vec "$STORE"/muse/dictionaries/*.txt "$STORE"/fasttext_cc/*.vec > "$STORE/SHA256SUMS"

# links where the notebooks look (Zhou: ./data/wiki.multi.xx.vec.txt and ./en-es.txt etc.)
Z="$EXT/Bias_in_Gendered_Languages"; mkdir -p "$Z/data"
for l in en es fr de; do ln -sf "$STORE/muse/wiki.multi.$l.vec" "$Z/data/wiki.multi.$l.vec.txt"; done
for d in en-es es-en es-fr; do ln -sf "$STORE/muse/dictionaries/$d.txt" "$Z/$d.txt"; done
G="$EXT/GG_Disentangling/data/embeddings"; mkdir -p "$G"
for l in es fr de it pl en; do ln -sf "$STORE/fasttext_cc/cc.$l.300.vec" "$G/cc.$l.300.vec"; done
echo "=== EMBEDDINGS done"
