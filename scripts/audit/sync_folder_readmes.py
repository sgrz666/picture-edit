#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ensure every folder in the repository has a proper README.md detailing its hierarchy.
If a directory already has a custom README.md, it is preserved.
If a directory has 目录说明.md but no README.md, README.md is created with full hierarchy.
"""
import os
import sys

ROOT = os.environ.get("PICEDIT_ROOT", "/home/shangguanrz/project/pic-edit")

def main():
    root_path = os.path.abspath(ROOT)
    print(f"Scanning directories under: {root_path}")
    
    created = []
    preserved = []
    
    for cur_dir, dirs, files in os.walk(root_path):
        # Exclude hidden directories, git, pycache, node_modules
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "node_modules"]
        
        rel = os.path.relpath(cur_dir, root_path)
        if rel == ".":
            continue
            
        doc_path = os.path.join(cur_dir, "目录说明.md")
        readme_path = os.path.join(cur_dir, "README.md")
        
        if os.path.exists(readme_path):
            preserved.append(rel)
        elif os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines = content.splitlines()
            if lines and lines[0].startswith("# "):
                lines[0] = f"# {rel}/ —— 目录层级与功能说明"
                new_content = "\n".join(lines)
            else:
                new_content = f"# {rel}/ —— 目录层级与功能说明\n\n" + content
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            created.append(rel)

    print(f"Preserved {len(preserved)} existing folder README.md files.")
    print(f"Created {len(created)} new folder README.md files from hierarchy docs.")

if __name__ == "__main__":
    main()
