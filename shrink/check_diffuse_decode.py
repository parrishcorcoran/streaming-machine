"""CHECK of the diffuse_plates family -- step 1, the decoder alone. Sees only each .bin file and a coder compiled
fresh from plates_rc.c (scratchpad/check_diffuse_rc.dylib). Never imports the machine or the original plates.
For each file: parse every header, every neuron's power of two, step term, line offset, the photon-coded stored numbers
(and in the dark-frame file, each neuron's constant), rebuild s_ij = max(0, (t_j + off_i - A) >> 1) and
w = v << s_ij saturated to 16 bits, and check that the parse used up exactly every byte of the file.
Writes the decoded plates (and constants) to scratchpad/check_dec_<tag>.npz for step 2 (the words)."""
import sys, os, struct, ctypes, json, numpy as np
SP = "/private/tmp/claude-501/-Users-abundancemachine/06869243-4b02-462e-91c7-0bac29e0c936/scratchpad"
lib = ctypes.CDLL(os.path.join(SP, "check_diffuse_rc.dylib"))
lib.plate_decode.restype = ctypes.c_int
lib.plate_decode.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]
NAMES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]

def dec(buf, R, C):
    b = np.frombuffer(buf, np.uint8).copy(); v = np.zeros((R, C), np.int32)
    assert lib.plate_decode(b.ctypes.data, len(buf), R, C, v.ctypes.data, 1) == 0
    return v

def read(path, dark):
    data = open(path, "rb").read(); p = 0
    hdr = "<BHHbhhII" if dark else "<BHHbhhI"; hs = struct.calcsize(hdr)
    out = {}; consts = {}; side = 0; coded = 0; cbytes = 0; nval = 0
    for _ in range(48):
        h = struct.unpack_from(hdr, data, p); p += hs; side += hs
        idx, R, C, A, tb, ob, nb = h[:7]
        e = np.frombuffer(data, np.int8, C, p).astype(np.int64); p += C
        t = np.frombuffer(data, np.uint8, C, p).astype(np.int64) + tb; p += C
        off = np.frombuffer(data, np.uint8, R, p).astype(np.int64) + ob; p += R
        side += 2 * C + R
        v = dec(data[p:p + nb], R, C).astype(np.int64); p += nb; coded += nb
        s = np.maximum(0, (t[None, :] + off[:, None] - A) >> 1)
        w = np.clip(np.left_shift(v, s), -32768, 32767).astype(np.int16)
        n = NAMES[idx]; out[n] = w; out[n + ".e"] = e; nval += R * C
        if dark:
            ncb = h[7]; c = dec(data[p:p + ncb], 1, C)[0].astype(np.int64); p += ncb; cbytes += ncb
            consts[n] = c
    assert p == len(data), (p, len(data))
    assert sorted(k for k in out if not k.endswith(".e")) == sorted(NAMES)
    return out, consts, dict(file_bytes=len(data), parsed_bytes=p, side_bytes=side, coded_bytes=coded,
                             const_bytes=cbytes, values=nval, MB=round(len(data) / 1e6, 3),
                             bits_per_value=round(8 * len(data) / nval, 4))

if __name__ == "__main__":
    jobs = [("v0", "diffuse_v0_plain_v12_A8.bin", False), ("v3", "diffuse_v3_carry_across_neurons_v12_A8.bin", False),
            ("v4", "diffuse_v4_carry_in_steps_v14_A6.bin", False), ("v9", "diffuse_v9_chained_dark_frame_v12_A6.bin", True),
            ("r1v14", "plates_v14_A6.bin", False)]
    sel = sys.argv[1:] or [j[0] for j in jobs]
    for tag, fn, dark in jobs:
        if tag not in sel: continue
        out, consts, info = read(os.path.join(SP, fn), dark)
        arrs = {k: v for k, v in out.items()}
        for n, c in consts.items(): arrs[n + ".c"] = c
        np.savez(os.path.join(SP, f"check_dec_{tag}.npz"), **arrs)
        info["file"] = fn; print(tag, json.dumps(info), flush=True)
