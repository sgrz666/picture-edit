"""Append the pictures/ + monitor/ bullets to the LoRA workspace README.

Idempotent: refuses to run twice. Verifies the pre-existing bytes are unchanged
before and after, so the hand-written part cannot be clobbered.
"""
import hashlib
import os
import sys

WS = "/data/shangguanrz/LoRA"
README = os.path.join(WS, "README.md")
APPEND = os.path.join(WS, "monitor_20261112/readme_append.md")

old = open(README, "rb").read()
add = open(APPEND, "rb").read()
if b"- `pictures/`" in old:
    print("ABORT: pictures bullet already present, nothing written")
    sys.exit(2)
if not add.endswith(b"\n"):
    add += b"\n"
if not old.endswith(b"\n"):
    old += b"\n"
new = old + add
with open(README, "wb") as f:
    f.write(new)
back = open(README, "rb").read()
assert back == new, "readback mismatch"
assert back.startswith(old), "original prefix changed"
print("README %d -> %d bytes ; prefix sha256 %s unchanged"
      % (len(old), len(back), hashlib.sha256(old).hexdigest()[:16]))
print("DONE")