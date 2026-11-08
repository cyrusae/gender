#!/usr/bin/env python3
"""
batch_ingest_symlink.py
Ingests PDFs from a source directory into a Zotero collection via zotero-cli,
then replaces the local file with an atomic symlink to the Zotero canonical storage.

Usage:
    python3 batch_ingest_symlink.py <folder_path> <collection_path> [tags] [--extra-collection <coll>]
"""

import os
import sys
import subprocess
import argparse
import time

def run_cmd(cmd):
    env = os.environ.copy()
    env["ZOTERO_LOCAL"] = "true"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    return res.returncode, res.stdout, res.stderr

def import_pdf(pdf_path, collections, tags, no_symlink=False):
    pdf_path = os.path.abspath(pdf_path)
    filename = os.path.basename(pdf_path)
    
    if os.path.islink(pdf_path):
        if no_symlink:
            os.remove(pdf_path)
            print(f"  [REMOVED SYMLINK] {filename} (direct Zotero mode)")
            return True, "removed_symlink"
        print(f"  [SKIP] Already a symlink: {filename}")
        return True, "already_symlink"

    print(f"  [INGESTING] {filename}...")
    coll_args = " ".join([f'-c "{c}"' for c in collections])
    cmd = f'zotero-cli add file --filepath "{pdf_path}" {coll_args} --create-collections --tags "{tags}"'
    code, stdout, stderr = run_cmd(cmd)
    
    if code != 0:
        print(f"  [ERROR] Failed to add {filename}: {stderr.strip() or stdout.strip()}")
        return False, "add_failed"

    item_key = None
    for line in stdout.splitlines():
        if "Item key:" in line:
            parts = line.split("`")
            if len(parts) >= 2:
                item_key = parts[1]
                break

    if not item_key:
        print(f"  [ERROR] Could not parse item key for {filename}: {stdout.strip()}")
        return False, "no_item_key"

    # Fetch local path
    cmd = f'zotero-cli path {item_key}'
    code, stdout, stderr = run_cmd(cmd)
    local_path = None
    for line in stdout.splitlines():
        if "- Local path:" in line:
            local_path = line.split("- Local path:")[1].strip().strip("`")
            break

    if not local_path or not os.path.exists(local_path):
        print(f"  [ERROR] Canonical storage file missing for {filename} (key={item_key}): {local_path}")
        return False, "storage_missing"

    if no_symlink:
        # In Direct Zotero Mode, remove staging file once confirmed in Zotero storage
        os.remove(pdf_path)
        print(f"  [SUCCESS DIRECT] {filename} (Key: {item_key}) -> stored in Zotero {local_path}")
    else:
        # Atomic symlink replacement
        tmp_link = pdf_path + ".tmp_symlink"
        if os.path.exists(tmp_link) or os.path.islink(tmp_link):
            os.remove(tmp_link)
        os.symlink(local_path, tmp_link)
        os.replace(tmp_link, pdf_path)
        print(f"  [SUCCESS SYMLINK] {filename} (Key: {item_key}) -> {local_path}")
    return True, item_key

def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs and symlink or store directly in Zotero")
    parser.add_argument("folder", help="Directory containing PDFs")
    parser.add_argument("collection", help="Zotero collection path (e.g. Projects/Name/01_Theme)")
    parser.add_argument("--tags", default="project:default", help="Comma-separated tags")
    parser.add_argument("-c", "--extra-collection", action="append", default=[], help="Additional collection to file into")
    parser.add_argument("--no-symlink", "--direct", action="store_true", help="Store directly in Zotero without creating local repo symlinks")
    args = parser.parse_args()

    collections = [args.collection] + args.extra_collection
    folder = os.path.abspath(args.folder)

    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a directory.")
        sys.exit(1)

    pdf_files = sorted([f for f in os.listdir(folder) if f.endswith(".pdf")])
    print(f"Found {len(pdf_files)} PDFs in {folder}")

    migrated = 0
    skipped = 0
    failed = 0

    for f in pdf_files:
        p = os.path.join(folder, f)
        ok, res = import_pdf(p, collections, args.tags, no_symlink=args.no_symlink)
        if ok:
            if res == "already_symlink":
                skipped += 1
            else:
                migrated += 1
        else:
            failed += 1
        time.sleep(0.2)

    print(f"\nDone: {migrated} ingested & symlinked, {skipped} skipped, {failed} failed.")

if __name__ == "__main__":
    main()
