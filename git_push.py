#!/usr/bin/env python3
"""
git_push.py - 변경사항 커밋 & 푸시
"""
import subprocess
import sys
import os

repo = r"C:\Foundry\Griptape\griptape-local-video-nodes"
os.chdir(repo)

commit_msg = sys.argv[1] if len(sys.argv) > 1 else "feat: Initial backup of Local Video Nodes (MiniMax-H3 LoRA + Wan2.1 + server)"

cmds = [
    ["git", "add", "-A"],
    ["git", "commit", "-m", commit_msg],
    ["git", "push", "origin", "main"],
]

for cmd in cmds:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            continue
        sys.exit(result.returncode)

print("\n✅ Push complete!")
