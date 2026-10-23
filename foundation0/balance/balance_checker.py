"""Independent rebuild of balance_count.py's main numbers, by a different road (a separate checker):
  - plates read with torch (safetensors' torch loader), not numpy
  - the power of two found by stepping, not by log2
  - sign and size: each place read by whole-number division (size // 2^p) % 2, ups and downs counted from the
    positive and the negative weights separately
  - signed digits: built digit by digit (odd: digit 2 - (n mod 4); then n = (n - digit) / 2), not by the bit trick
  - beyond 3 widths decided in whole numbers: net^2 > 9 x (sum of squares), (ups - downs)^2 > 9 x (ups + downs)
Compares, for every plate and both steps: the power, the ups and downs at every place in both forms, the plate's net,
and the shares of neurons and lines whose net is beyond 3 widths, and of (neuron, place) cells beyond 3 widths.

    python3 balance_checker.py [sizes=70m,160m]
"""
import os, sys, json, torch
from safetensors.torch import safe_open

A = dict(a.split("=") for a in sys.argv[1:])
SIZES = A.get("sizes", "70m,160m").split(",")
HERE = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(os.path.dirname(HERE), "models")
STEPS = ("step0", "step143000")
torch.set_grad_enabled(False)


def plate_list(L):
    out = []
    for l in range(L):
        p = f"gpt_neox.layers.{l}."
        out += [(f"qkv.{l}", p + "attention.query_key_value.weight", 1), (f"attn_out.{l}", p + "attention.dense.weight", 1),
                (f"mlp_in.{l}", p + "mlp.dense_h_to_4h.weight", 1), (f"mlp_out.{l}", p + "mlp.dense_4h_to_h.weight", 1)]
    return out + [("words_in", "gpt_neox.embed_in.weight", 0), ("words_out", "embed_out.weight", 1)]


def power(top):
    e = 0
    while top / 2.0 ** e > 32767: e += 1
    while top / 2.0 ** (e - 1) <= 32767: e -= 1
    return e


def digits_ss(q):
    """per place: (+1 at that place, -1 at that place) as int8 tensors, sign and size"""
    n = q.abs(); pos = (q > 0).to(torch.int8); neg = (q < 0).to(torch.int8)
    for p in range(15):
        b = torch.div(n, 2 ** p, rounding_mode="floor").remainder(2).to(torch.int8)
        yield b * pos, b * neg
    assert int(torch.div(n, 2 ** 15, rounding_mode="floor").max()) == 0


def digits_sd(q):
    n = q.abs(); s = torch.sign(q).to(torch.int8)
    for p in range(16):
        odd = n.remainder(2) == 1
        d = torch.where(odd, 2 - n.remainder(4), torch.zeros_like(n))
        n = torch.div(n - d, 2, rounding_mode="floor")
        dv = d.to(torch.int8) * s
        yield (dv == 1).to(torch.int8), (dv == -1).to(torch.int8)
    assert int(n.abs().max()) == 0


def share3(net, sq):
    return float((net * net > 9 * sq).double().mean())


bad = 0; checked = 0
for SIZE in SIZES:
    cfg = json.load(open(os.path.join(MD, f"pythia-{SIZE}-step143000", "config.json")))
    ref = {r["plate"]: r for r in json.load(open(os.path.join(HERE, f"balance_{SIZE}.json")))}
    F = {st: safe_open(os.path.join(MD, f"pythia-{SIZE}-{st}", "model.safetensors"), "pt") for st in STEPS}
    for name, key, nax in plate_list(cfg["num_hidden_layers"]):
        lax = 1 - nax
        top = max(float(F[st].get_tensor(key).float().abs().max()) for st in STEPS)
        e = power(top); R = ref[name]
        if e != R["power"]: print(f"MISMATCH {SIZE} {name} power {e} vs {R['power']}"); bad += 1
        for st in STEPS:
            q = torch.round(F[st].get_tensor(key).float() / 2.0 ** e).to(torch.int32)   # torch.round: half to even
            S = R["steps"][st]
            q64 = q.to(torch.int64)
            net = int(q64.sum())
            nn, nsq = q64.sum(nax), (q64 * q64).sum(nax); ln, lsq = q64.sum(lax), (q64 * q64).sum(lax)
            del q64
            mine = dict(neuron_net3=share3(nn, nsq), line_net3=share3(ln, lsq))
            for form, gen in (("ss", digits_ss), ("sd", digits_sd)):
                ups, dns = [], []; c3 = cl3 = live_n = live_l = 0
                for u, d in gen(q):
                    ups.append(int(u.sum(dtype=torch.int64))); dns.append(int(d.sum(dtype=torch.int64)))
                    dn_ = (u - d).sum(nax, dtype=torch.int64); tn = (u + d).sum(nax, dtype=torch.int64)
                    dl_ = (u - d).sum(lax, dtype=torch.int64); tl = (u + d).sum(lax, dtype=torch.int64)
                    c3 += int(((dn_ * dn_ > 9 * tn) & (tn > 0)).sum()); live_n += int((tn > 0).sum())
                    cl3 += int(((dl_ * dl_ > 9 * tl) & (tl > 0)).sum()); live_l += int((tl > 0).sum())
                a = S[form]["plate"]
                checked += 1
                same = (ups == a["ups"] and dns == a["downs"] and net == a["plate_net"]
                        and abs(mine["neuron_net3"] - a["neuron_net3"]) < 1e-12 and abs(mine["line_net3"] - a["line_net3"]) < 1e-12
                        and abs(c3 / max(live_n, 1) - a["neuron_cell3"]) < 1e-12 and abs(cl3 / max(live_l, 1) - a["line_cell3"]) < 1e-12)
                if not same:
                    bad += 1
                    print(f"MISMATCH {SIZE} {name} {st} {form}: ups {ups == a['ups']} downs {dns == a['downs']} net {net} vs {a['plate_net']} "
                          f"neuron {mine['neuron_net3']} vs {a['neuron_net3']} line {mine['line_net3']} vs {a['line_net3']} "
                          f"cells {c3 / max(live_n, 1)} vs {a['neuron_cell3']}, {cl3 / max(live_l, 1)} vs {a['line_cell3']}")
            del q
        print(f"{SIZE} {name}: checked", flush=True)
print(f"\n{checked} (plate, step, form) records checked, {bad} mismatches")
