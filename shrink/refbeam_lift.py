"""REFERENCE BEAM family: integer lifting lenses (exact, invertible in whole numbers: subtract, shift, add).
  The 2-point step on a pair (a, b):   d = a - b ;  s = b + (d >> 1)        (s is floor((a+b)/2), d the difference)
  undone:                              b = s - (d >> 1) ;  a = d + b
  haar(X, levels, axis)  : the step on the low band again and again (each level halves the low band)
  wht(X, levels, axis)   : the step on every band at every level (the Walsh-Hadamard lens, as lifting)
Both work along the last axis of X (pass X.T for words). The output keeps positions: low bands first.
"""
import numpy as np

def _step(Y):                                        # pairs along the last axis
    a, b = Y[..., 0::2], Y[..., 1::2]; d = a - b; s = b + (d >> 1)
    return s, d

def _unstep(s, d):
    b = s - (d >> 1); a = d + b
    out = np.empty(s.shape[:-1] + (2 * s.shape[-1],), np.int64); out[..., 0::2] = a; out[..., 1::2] = b
    return out

def haar(X, levels):
    X = np.array(X, dtype=np.int64); n = X.shape[-1]
    for _ in range(levels):
        s, d = _step(X[..., :n]); X[..., :n // 2] = s; X[..., n // 2:n] = d; n //= 2
    return X

def ihaar(X, levels):
    X = np.array(X, dtype=np.int64); n = X.shape[-1] >> levels
    for _ in range(levels):
        X[..., :2 * n] = _unstep(X[..., :n], X[..., n:2 * n]); n *= 2
    return X

def wht(X, levels):
    X = np.array(X, dtype=np.int64); L = X.shape[-1]; lead = X.shape[:-1]
    for lv in range(levels):
        g = 1 << lv; Y = X.reshape(lead + (g, L // g)); s, d = _step(Y)
        X = np.concatenate([s, d], axis=-1).reshape(lead + (L,))
    return X

def iwht(X, levels):
    X = np.array(X, dtype=np.int64); L = X.shape[-1]; lead = X.shape[:-1]
    for lv in range(levels - 1, -1, -1):
        g = 1 << lv; Y = X.reshape(lead + (g, L // g)); h = L // g // 2
        X = _unstep(Y[..., :h], Y[..., h:]).reshape(lead + (L,))
    return X

if __name__ == "__main__":
    rng = np.random.default_rng(1); X = rng.integers(-100000, 100000, size=(7, 768))
    for lv in range(1, 9):
        assert np.array_equal(ihaar(haar(X, lv), lv), X); assert np.array_equal(iwht(wht(X, lv), lv), X)
    print("lifting exact both ways")
