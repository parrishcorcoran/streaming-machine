"""Fetch the Pythia plates: the random start (step0), the finished plates (step143000), and for pythia-70m three
steps between. Each is written as models/<name>-<step>/model.safetensors in fp16, the form Pythia's own files hold
(the fp32 model.safetensors on the Hub is the fp16 values widened: checked equal, every tensor, on pythia-70m
step143000). Only the plates are kept (no causal-mask or turn-rate buffers). Each file is read back after writing
and checked equal to the download before the download is deleted.

    python3 fetch.py
"""
import os, shutil, torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
WANT = [("pythia-14m", "step0"), ("pythia-14m", "step143000"),
        ("pythia-70m", "step0"), ("pythia-70m", "step1000"), ("pythia-70m", "step10000"), ("pythia-70m", "step50000"),
        ("pythia-70m", "step143000"),
        ("pythia-160m", "step0"), ("pythia-160m", "step143000")]
SKIP = ("attention.bias", "attention.masked_bias", "rotary_emb.inv_freq")

def keep(sd):
    return {k: v.contiguous() for k, v in sd.items() if not k.endswith(SKIP)}

for name, step in WANT:
    out = os.path.join(D, f"{name}-{step}"); dst = os.path.join(out, "model.safetensors")
    hf_hub_download(f"EleutherAI/{name}", "config.json", revision=step, local_dir=out)
    if os.path.exists(dst) and not os.path.exists(os.path.join(out, "pytorch_model.bin")):
        with safe_open(dst, "pt") as f:
            dt = {f.get_slice(k).get_dtype() for k in f.keys()}
        if dt == {"F16"}:
            print(f"{name} {step}: already fp16, {os.path.getsize(dst):,} bytes"); continue
    src = os.path.join(out, "pytorch_model.bin")
    if not os.path.exists(src):
        hf_hub_download(f"EleutherAI/{name}", "pytorch_model.bin", revision=step, local_dir=out)
    sd = keep(torch.load(src, map_location="cpu", weights_only=True))
    assert {v.dtype for v in sd.values()} == {torch.float16}, {v.dtype for v in sd.values()}
    tmp = dst + ".tmp"; save_file(sd, tmp, metadata={"format": "pt"})
    with safe_open(tmp, "pt") as f:
        assert set(f.keys()) == set(sd)
        for k in sd: assert torch.equal(f.get_tensor(k), sd[k]), k
    if os.path.exists(dst):                                    # an fp32 copy from the Hub: check it is the same values
        with safe_open(dst, "pt") as f:
            for k in sd: assert torch.equal(f.get_tensor(k).half().float(), f.get_tensor(k).float()) and torch.equal(f.get_tensor(k).half(), sd[k]), k
    os.replace(tmp, dst); os.remove(src)
    print(f"{name} {step}: {len(sd)} plates, {sum(v.numel() for v in sd.values()):,} values, {os.path.getsize(dst):,} bytes")
    del sd

tk = os.path.join(D, "tokenizer")
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    hf_hub_download("EleutherAI/pythia-70m", f, revision="step143000", local_dir=tk)
for root, dirs, _ in os.walk(D):
    if ".cache" in dirs: shutil.rmtree(os.path.join(root, ".cache"))
print("total", sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(D) for f in fs) / 1e9, "GB")
