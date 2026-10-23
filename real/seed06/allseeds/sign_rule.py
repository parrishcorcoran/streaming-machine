"""Check the sign rule of each CPU draw against real torch on this Mac.
A bf16 (8-bit uniforms), B fp16 (11-bit), C fp32 (24-bit), D fp64 (2 words a value, 53-bit), all in blocks of 16
(value j and j+8 share the angle of value j+8's uniform). E: the one-at-a-time draw (non-contiguous or < 16 values):
4 words per 2 values, angle from the first 64-bit uniform. Sign of cos = top XOR next bit of the angle; sin = top bit."""
import numpy as np, torch
def words(seed, n):
    bg = np.random.MT19937(); bg._legacy_seeding(seed); return bg.random_raw(n).astype(np.uint64)
def rule(seed, conv, n=2048):
    if conv in "ABC":
        hi = {"A": 7, "B": 10, "C": 23}[conv]; w = words(seed, n); ang = w
    elif conv == "D":
        w = words(seed, 2 * n); ang = w[0::2]; hi = 20            # first word of each pair is the high word
    if conv in "ABCD":
        s = np.zeros(n, bool)
        for b in range(0, n, 16):
            a = ang[b + 8:b + 16]; t = (a >> np.uint64(hi)) & np.uint64(1); u = (a >> np.uint64(hi - 1)) & np.uint64(1)
            s[b:b + 8] = (t ^ u).astype(bool); s[b + 8:b + 16] = t.astype(bool)
        return s
    w = words(seed, 2 * n); a = w[0::4]; t = (a >> np.uint64(20)) & np.uint64(1); u = (a >> np.uint64(19)) & np.uint64(1)
    s = np.zeros(n, bool); s[0::2] = (t ^ u).astype(bool); s[1::2] = t.astype(bool); return s
def real(seed, conv, n=2048):
    torch.manual_seed(seed)
    if conv == "E": x = torch.empty(2 * n, dtype=torch.float64)[::2]; x.normal_(); return x.numpy() < 0, x.numpy() == 0
    dt = {"A": torch.bfloat16, "B": torch.float16, "C": torch.float32, "D": torch.float64}[conv]
    x = torch.empty(n, dtype=dt).normal_().float().numpy(); return x < 0, x == 0
for conv in "ABCDE":
    out = []
    for seed in (0, 1, 42, 123456789, 3289271898, 4294967295):
        neg, zero = real(seed, conv); r = rule(seed, conv); ok = (neg == r) | zero
        out.append(f"{100*ok.mean():.2f}%")
    print(conv, "sign rule matches real torch:", " ".join(out), flush=True)
