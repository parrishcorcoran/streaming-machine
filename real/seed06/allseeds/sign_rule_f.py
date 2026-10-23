"""Uniform-family and truncated-bell draws: sign of value i = NOT top bit of its own uniform (one word a value for
bf16/fp16/fp32 at bits 7/10/23; fp64 two words a value, high word first, bit 20). Checked on real torch."""
import numpy as np, torch
def words(seed, n):
    bg = np.random.MT19937(); bg._legacy_seeding(seed); return bg.random_raw(n).astype(np.uint64)
def rule(seed, dt, n=2048):
    if dt == torch.float64: a = words(seed, 2 * n)[0::2]; hi = 20
    else: a = words(seed, n); hi = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}[dt]
    return ((a >> np.uint64(hi)) & np.uint64(1)) == 0
for dt in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
    for how in ("uniform_(-1,1)", "trunc_normal_(std=0.02)", "kaiming_uniform_"):
        res = []
        for seed in (0, 1, 42, 123456789, 3289271898, 4294967295):
            torch.manual_seed(seed); x = torch.empty(2048, dtype=dt)
            if how.startswith("uniform"): x.uniform_(-1, 1)
            elif how.startswith("trunc"): torch.nn.init.trunc_normal_(x, std=0.02)
            else: x = torch.empty(16, 128, dtype=dt); torch.nn.init.kaiming_uniform_(x, a=5 ** 0.5); x = x.reshape(-1)
            v = x.double().numpy(); ok = ((v < 0) == rule(seed, dt)) | (v == 0); res.append(f"{100*ok.mean():.2f}%")
        print(f"{str(dt):15s} {how:24s}", " ".join(res))
