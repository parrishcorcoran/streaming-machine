"""GPT-2 in base 2, the 12 layers written out as photons only: no zeros. Every weight is 15 places, most significant
first; a photon going up is ` 1`, a photon going down is `-1`, a dark place is blank. Weights separated by `|`.
    python3 write_ones.py            -> gpt2_base2_layers.md
"""
import numpy as np
M = np.load("base2_gpt2.npz"); PER_LINE = 4
def photons(v):                                        # a whole number as its photons only, signed, top place first
    v = int(v); s = "-1" if v < 0 else " 1"; return "".join(s if c == "1" else "  " for c in bin(abs(v))[2:])
names = [("qkv", "queries, keys, values"), ("proj", "the arrival's result"), ("fc", "the neurons in"), ("mproj", "the neurons out")]
with open("gpt2_base2_layers.md", "wb") as f:
    f.write(("# GPT-2 in base 2: the 12 layers, every weight as its photons\n\n"
             "No zeros. Each weight is 15 places, most significant first: ` 1` a photon going up, `-1` a photon going down,\n"
             "blank a dark place; weights separated by `|`, one for each line of light coming in, 4 to a row.\n"
             "Each neuron has its power of two (every one of its weights is x 2^power) and its offset, written the same way.\n"
             "The banks in each layer are in the order the light meets them.\n").encode())
    for l in range(12):
        f.write(f"\n## layer {l}\n".encode())
        for key, what in names:
            w, e, b = M[f"{key}.{l}.w"], M[f"{key}.{l}.e"], M[f"{key}.{l}.b"]; nin, nout = w.shape
            W = w.T.astype(np.int32); mag = np.abs(W); neg = W < 0; assert mag.max() < 32768
            ch = np.full((nout, nin, 31), 32, np.uint8)
            for k in range(15):
                lit = ((mag >> (14 - k)) & 1) == 1
                ch[:, :, 2 * k] = np.where(lit & neg, 45, 32); ch[:, :, 2 * k + 1] = np.where(lit, 49, 32)
            ch[:, :, 30] = 124; ch[:, PER_LINE - 1::PER_LINE, 30] = 10
            f.write(f"\n### layer {l} · {what} · {nin:,} in -> {nout:,} out\n".encode())
            for j in range(nout):
                f.write(f"\nneuron {j} · power `{photons(e[j])}` · offset `{photons(b[j])}`\n```\n".encode())
                f.write(ch[j].tobytes()); f.write(b"```\n")
        print(f"  layer {l} written", flush=True)
