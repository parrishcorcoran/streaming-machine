"""How much the tiny layer changes when one word is added. Tiny layer for the prompt (J_S: old pieces -> top) and for
the prompt + its next word (J_S+1: old pieces + the new one -> the new top). Compare J_S+1's part on the old pieces
with J_S, and size the new piece's own part.
    python3 tiny_change.py
"""
import torch
exec(open("tiny_qwen.py").read().split("from transformers import AutoTokenizer; tok")[0])
for r in refs[:3]:
    ids = r["ids"]; S = len(ids)
    J0, o0, _, _ = tiny(ids); J1, o1, _, _ = tiny(ids + [r["ref"][0]])
    old, new = J1[:, :S * 1024], J1[:, S * 1024:]
    print(f"  {r['prompt']!r:28s} stack part changed by {100 * (old - J0).norm() / J0.norm():5.1f}% of its size;"
          f"  the new piece's own part is {100 * new.norm() / J1.norm():5.1f}% of the new layer;  offsets changed {100 * (o1 - o0).norm() / o0.norm():5.1f}%")
