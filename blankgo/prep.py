"""The text and the word table for the tiny models (run once, with a Python that has pyarrow):
    ~/Exponential-Inference/.venv/bin/python prep.py

  - text: wikitext-2 raw, read from the Hugging Face cache (arrow files). The building part is its train split; the
    held-out part is its validation split. Nothing of the held-out part is used to choose anything.
  - words: GPT-2's own tokenizer. The word table is limited to the 8,192 words (GPT-2 tokens) that are most common in
    the BUILDING part; every other word maps to one more row, row 8,192 (the rest). 8,193 rows in all.
    Ties in the count are broken by the GPT-2 id (smaller first), so the table is fixed by the text alone.
  - writes data.npz: build (uint16 rows), held (uint16 rows), gpt2_id (the GPT-2 id of each of the 8,192 rows),
    count (each row's count in the building part).

PREDICTION (written before the run): the building part is about 2.4-2.6 million GPT-2 tokens, the held-out part about
250 thousand; the 8,192 most common words cover 93-97% of the building part's tokens and slightly less of the
held-out part (the rest row takes 3-7%).
"""
import os, json, numpy as np, pyarrow as pa
from transformers import GPT2TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser("~/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3")
tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.model_max_length = 10 ** 9

def read(split):
    t = pa.ipc.open_stream(open(os.path.join(SRC, f"wikitext-{split}.arrow"), "rb")).read_all()
    return "".join(t.column("text").to_pylist())

ids = {s: np.array(tok(read(s)).input_ids, dtype=np.int64) for s in ("train", "validation")}
cnt = np.bincount(ids["train"], minlength=len(tok))
order = np.lexsort((np.arange(len(cnt)), -cnt))[:8192]          # most common first; ties by the smaller id
row = np.full(len(tok), 8192, dtype=np.int64); row[order] = np.arange(8192)
build, held = row[ids["train"]].astype(np.uint16), row[ids["validation"]].astype(np.uint16)
np.savez(os.path.join(HERE, "data.npz"), build=build, held=held, gpt2_id=order, count=cnt[order])
rep = dict(build_tokens=int(len(build)), held_tokens=int(len(held)),
           build_covered=float((build < 8192).mean()), held_covered=float((held < 8192).mean()),
           rows=8193, least_common_kept=int(cnt[order[-1]]))
json.dump(rep, open(os.path.join(HERE, "prep.json"), "w"), indent=1)
print(rep)
