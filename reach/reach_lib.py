"""reach/reach_lib.py: the bank's own rounding as a bucket of piles (the same functions reach/reach.py uses)."""
import numpy as np
import fastm as F

def bucket_gauged(P, F_):
    """kept = power(P, F) = rshift(P x EXP2[f], 16 - I). Returns (lo, hi) of the piles with the same kept value."""
    I, f = np.right_shift(F_, 16), np.bitwise_and(F_, 0xFFFF); g = F.EXP2[f]; u = 16 - I
    s = u > 0; us = np.where(s, u, 1); h = np.left_shift(np.int64(1), us - 1)
    k = np.right_shift(P * g + h, us); A = np.left_shift(k, us) - h; Bv = A + np.left_shift(np.int64(1), us) - 1
    return np.where(s, -((-A) // g), P), np.where(s, Bv // g, P)
def bucket_plain(P, sh):
    """kept = rshift(P, sh). Returns (lo, hi)."""
    s = sh > 0; ss = np.where(s, sh, 1); h = np.left_shift(np.int64(1), ss - 1)
    k = np.right_shift(P + h, ss); lo = np.left_shift(k, ss) - h; hi = lo + np.left_shift(np.int64(1), ss) - 1
    return np.where(s, lo, P), np.where(s, hi, P)
def margins(P, kept_fn, lo, hi):
    """m = min(P - lo, hi - P); the bucket checked exactly: the same kept value at lo and hi, a different one past them."""
    K = kept_fn(P); assert np.array_equal(kept_fn(lo), K) and np.array_equal(kept_fn(hi), K)
    assert (kept_fn(lo - 1) != K).all() and (kept_fn(hi + 1) != K).all()
    return np.minimum(P - lo, hi - P)
