#!/usr/bin/env python3
"""
pin_collection_guides.py
Pins a scope and reading guide item titled `📌 00 Scope & Reading Guide: <Name>`
at the top of a Zotero collection and attaches a formatted Markdown child note.

Usage:
    python3 pin_collection_guides.py <collection_path> <title> --note-file <path_to_markdown>
"""

import subprocess
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Pin scope guide and note in a Zotero collection")
    parser.add_argument("collection", help="Zotero collection path (e.g. Projects/Name/01_Theme)")
    parser.add_argument("title", help="Guide title (e.g. '01 Human Psycholinguistics')")
    parser.add_argument("--note-file", required=True, help="Markdown file containing scope/triage text")
    parser.add_argument("--tags", default="role:background-only,#nocite,type:reading-guide", help="Tags for guide item")
    args = parser.parse_args()

    env = os.environ.copy()
    env["ZOTERO_LOCAL"] = "true"

    if not os.path.exists(args.note_file):
        print(f"Error: {args.note_file} not found.")
        sys.exit(1)

    with open(args.note_file, "r") as f:
        note_text = f.read()

    full_title = f"📌 00 Scope & Reading Guide: {args.title}"
    print(f"Creating guide item in {args.collection}: {full_title}...")

    # 1. Add file or report item
    add_cmd = [
        "zotero-cli", "add", "file",
        "--filepath", os.path.abspath(args.note_file),
        "-c", args.collection,
        "--create-collections",
        "--tags", args.tags,
        "--item-type", "report",
        "--title", full_title
    ]
    res = subprocess.run(add_cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"Error creating guide item: {res.stderr}")
        sys.exit(1)

    item_key = None
    for line in res.stdout.splitlines():
        if "Item key:" in line:
            parts = line.split("`")
            if len(parts) >= 2:
                item_key = parts[1]
                break

    if not item_key:
        print(f"Failed to parse item key from: {res.stdout}")
        sys.exit(1)

    # 2. Attach rich child note
    note_cmd = [
        "zotero-cli", "notes", "create",
        "--item-key", item_key,
        "--title", f"Scope & Guiding Logic: {args.title}",
        "--text", note_text
    ]
    subprocess.run(note_cmd, capture_output=True, text=True, env=env)

    # 3. Mark extra field
    edit_cmd = [
        "zotero-cli", "edit", item_key,
        "--extra", "tex.skipbib: true\\nkeywords: nocite, reading-guide"
    ]
    subprocess.run(edit_cmd, capture_output=True, text=True, env=env)

    print(f"Successfully pinned guide {full_title} (Key: {item_key}) with rich child note.")

if __name__ == "__main__":
    main()
