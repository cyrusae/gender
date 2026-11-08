#!/usr/bin/env python3
"""
enrich_zotero_metadata.py
Inspects items in Zotero with raw filename titles (e.g. `[Author Year] Title.pdf`),
parses them into structured fields (Title, Short Title, Date, Creators),
and enriches with full author names from SUMMARIES.md if available.

Usage:
    python3 enrich_zotero_metadata.py [--summaries <path_to_summaries.md>] [--limit <limit>]
"""

import subprocess
import json
import os
import re
import time
import argparse

def load_summaries(summaries_path):
    citations_by_file = {}
    if not summaries_path or not os.path.exists(summaries_path):
        return citations_by_file
    with open(summaries_path, "r") as f:
        text = f.read()
    entries = re.split(r"\n---\n+", text)
    for entry in entries:
        file_match = re.search(r"\*\*File:\*\*\s*`?(?:.*?\/)?(\[.*?\]\s*.*?\.pdf)", entry)
        cit_match = re.search(r"\*\*Citation:\*\*\s*(.*?)(?:\n\n|\n\*\*|\Z)", entry, re.DOTALL)
        if file_match and cit_match:
            fname = file_match.group(1).strip()
            cit = cit_match.group(1).strip().replace("\n", " ")
            citations_by_file[fname] = cit
    return citations_by_file

def parse_citation_authors(cit):
    m = re.match(r"^(.*?)\s*\(\d{4}[a-z]?\)\.", cit)
    if not m:
        return None
    author_str = m.group(1).strip()
    author_chunks = re.split(r";\s*|\s+and\s+", author_str)
    creators = []
    for chunk in author_chunks:
        chunk = chunk.strip().rstrip(",")
        if not chunk:
            continue
        if "," in chunk:
            parts = chunk.split(",", 1)
            creators.append({"creatorType": "author", "lastName": parts[0].strip(), "firstName": parts[1].strip()})
        else:
            creators.append({"creatorType": "author", "lastName": chunk.strip()})
    return creators if creators else None

def main():
    parser = argparse.ArgumentParser(description="Enrich raw Zotero items with clean metadata")
    parser.add_argument("--summaries", default="sources/SUMMARIES.md", help="Path to project SUMMARIES.md")
    parser.add_argument("--limit", type=int, default=300, help="Number of recent items to inspect")
    args = parser.parse_args()

    env = os.environ.copy()
    env["ZOTERO_LOCAL"] = "true"

    citations = load_summaries(args.summaries)
    if citations:
        print(f"Loaded {len(citations)} full citations from {args.summaries}")

    res = subprocess.run(["zotero-cli", "--json", "get", "recent", "--limit", str(args.limit)], capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print("Failed to fetch recent items from Zotero:", res.stderr)
        return

    data = json.loads(res.stdout)
    items = data["data"]["items"]
    print(f"Inspecting {len(items)} items in Zotero library...")

    pattern = re.compile(r"^\[(.*?) (\d{4}[a-z]?)\] (.*?)\.pdf$")
    updated_count = 0
    skipped_count = 0

    for item in items:
        raw_title = item.get("title", "")
        item_key = item.get("key", "")

        if not (raw_title.startswith("[") and raw_title.endswith(".pdf")):
            skipped_count += 1
            continue

        m = pattern.match(raw_title)
        if not m:
            continue

        author_str, year, full_title = m.groups()

        # Clean short title
        if " - " in full_title:
            short_title = full_title.split(" - ", 1)[0].strip()
        elif ":" in full_title:
            short_title = full_title.split(":", 1)[0].strip()
        else:
            short_title = full_title

        # Authors
        creators = None
        if raw_title in citations:
            creators = parse_citation_authors(citations[raw_title])

        if not creators:
            cleaned_authors = author_str.replace(" et al.", "").strip()
            authors = re.split(r",\s*|\s*&\s*", cleaned_authors)
            creators = [{"creatorType": "author", "lastName": a.strip()} for a in authors if a.strip()]

        edit_cmd = [
            "zotero-cli", "edit", item_key,
            "--title", full_title,
            "--short-title", short_title,
            "--date", year,
            "--creators", json.dumps(creators)
        ]

        edit_res = subprocess.run(edit_cmd, capture_output=True, text=True, env=env)
        if edit_res.returncode == 0:
            updated_count += 1
            print(f"[{updated_count}] Enriched {item_key}: {short_title} ({year})")
        else:
            print(f"Error enriching {item_key}: {edit_res.stderr.strip()}")

        time.sleep(0.05)

    print(f"\nCompleted: {updated_count} items enriched, {skipped_count} already structured.")

if __name__ == "__main__":
    main()
