"""The parts each stored weight holds, for one row (Qwen3-0.6B layer 0 up_proj row 0), bf16 / Q8_0 / Q4_0."""
import math, numpy as np
exec(open("circles.py").read().split("# --- base-2 circles")[0])   # loads row, q8_0, q4_0
out = open("parts.tsv", "w")
out.write("i\tbf16_value\tbf16_sign\tbf16_exponent\tbf16_mantissa(1.m)\tq8_value\tq8_shared_scale(32)\tq8_code\tq4_value\tq4_shared_scale(32)\tq4_code\n")
def scales(x, kind):
    s = np.empty_like(x)
    for b in range(0, x.size, 32):
        v = x[b:b+32]
        if kind == 8: d = np.float32(np.max(np.abs(v)) / np.float32(127))
        else: d = np.float32(v[int(np.argmax(np.abs(v)))] / np.float32(-8))
        s[b:b+32] = np.float32(np.float16(d))
    return s
s8, s4 = scales(row, 8), scales(row, 4)
for i, v in enumerate(row.astype(np.float64)):
    fr, e = math.frexp(abs(v)); man = fr * 2; e -= 1          # |v| = man * 2^e, man in [1,2)
    sign = "+" if v > 0 else "-"
    out.write(f"{i}\t{v:.9g}\t{sign}\t{e}\t{man:.7f}\t{q8[i]:.9g}\t{s8[i]:.6g}\t{c8[i]}\t{q4[i]:.9g}\t{s4[i]:.6g}\t{c4[i]-8}\n")
out.close()
