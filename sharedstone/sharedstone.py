# The stone is what is shared. Each gate: line up all its weights in base 2 (16 places); any place where EVERY weight
# has a 1 (shared 1) or EVERY weight has a 0 (shared 0) is stone. The tip of each weight is its unshared places.
# Exact: pile = stone x (sum of light) + sum(light x tip). Same on the light side: each line across all tokens.
import numpy as np
M = np.load("../base2_gpt2.npz")
rng = np.random.default_rng(0)
def shared_places(U):                       # U: (n, groups) as 16-place patterns; places all-1 or all-0 per group
    ones = np.bitwise_and.reduce(U, axis=0); zeros = np.bitwise_and.reduce(~U & 0xFFFF, axis=0)
    mask = (ones | zeros) & 0xFFFF
    return mask, np.array([bin(m).count("1") for m in mask])
def bits(W):  return W.astype(np.int64) & 0xFFFF   # 16 places, as written
print("WEIGHT SIDE: each gate's weights lined up in base 2 (16 places). Places shared by every weight = the stone.")
print(f"{'plate':10s} {'weights/gate':>12s} {'shared places (of 16)':>22s} {'tip places':>10s} {'n/a tips':>9s}")
tot_tip=tot_n=0
for kind in ["qkv","proj","fc","mproj"]:
    for l in [0,6,11]:
        W = M[f"{kind}.{l}.w"]; U = bits(W)
        mask, sh = shared_places(U)
        stone = (U[0] & mask)                                   # the gate's stone pattern
        tip = U & ~mask & 0xFFFF
        if l==6 and kind=="fc":                                 # exactness check on real light
            x = np.load("../groups/light_gpt2/fc.npy", mmap_mode="r")[6, :5].astype(np.int64)
            Ws = W.astype(np.int64); st = np.where(stone>=32768, stone-65536, stone)  # signed stone
            # w = stone + tip (as signed), so pile = stone*sum(x) + x.tip_signed
            tipS = Ws - st[None,:]
            assert np.array_equal(x@Ws, x.sum(1)[:,None]*st[None,:] + x@tipS)
        print(f"{kind}.{l:<6d} {W.shape[0]:12d} {sh.mean():22.2f} {16-sh.mean():10.2f} {np.mean(tip==0)*100:8.2f}%")
print("  (exactness pile = stone x light sum + light x tips: checked on fc.6 with real light, holds)")
print()
print("WHY: share of weights with a 1 at each place (fc.6, all gates), top place first:")
U = bits(M["fc.6.w"]); print("  " + " ".join(f"{np.mean((U>>p)&1)*100:3.0f}" for p in range(15,-1,-1)))
print()
print("RANDOM GROUPS of weights inside a gate (random only): shared places as the group shrinks (fc.6):")
W = bits(M["fc.6.w"])
for g in [2,4,8,16,64,768]:
    perm = rng.permutation(768); G = W[perm[: (768//g)*g]].reshape(768//g, g, -1)
    ones = np.bitwise_and.reduce(G, axis=1); zeros = np.bitwise_and.reduce(~G & 0xFFFF, axis=1)
    m = (ones|zeros)&0xFFFF; sh = np.vectorize(lambda v: bin(int(v)).count("1"))(m[:, :256]).mean()
    # control: same values shuffled across the whole plate (distribution kept, structure gone)
    Wc = W.reshape(-1)[rng.permutation(W.size)].reshape(W.shape)[perm[:(768//g)*g]].reshape(768//g,g,-1)
    mc = (np.bitwise_and.reduce(Wc,axis=1)|np.bitwise_and.reduce(~Wc&0xFFFF,axis=1))&0xFFFF
    shc = np.vectorize(lambda v: bin(int(v)).count("1"))(mc[:, :256]).mean()
    print(f"  groups of {g:3d}: shared {sh:5.2f} of 16 places -> tip {16-sh:5.2f} places | shuffled control {shc:5.2f}")
print()
print("LIGHT SIDE: each line's light across all 647 tokens after the first, lined up in base 2. Shared = the light's stone.")
for kind in ["qkv","fc","mproj"]:
    X = np.load(f"../groups/light_gpt2/{kind}.npy", mmap_mode="r")
    for l in [0,6,11]:
        U = np.asarray(X[l,1:]).astype(np.int64) & 0xFFFF
        mask, sh = shared_places(U)
        print(f"  {kind}.{l:<3d} light: shared {sh.mean():5.2f} of 16 places per line")
