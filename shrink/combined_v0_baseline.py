"""COMBINED v0 -- the starting point, re-judged with everything counted. Round one's 81.7 MB (plates v12 A=8 56.74 MB +
tables c=34 24.96 MB) left out the numbers that were not developed: the 48 plates' biases, the head's bias, the
position table, the arithmetic tables. They are coded here exactly (decoded and compared equal) and added.
Prediction before running: 30/30 (the same machine round one judged), total = 81.7 + ~1.3 MB.
Also: the position table coded as is vs as changes from the position before (a coder choice, both exact)."""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
P = tb.M["wpe"].astype(np.int64); dP = P.copy(); dP[1:] = P[1:] - P[:-1]
print("position table as is:", len(rc(P)), "bytes;  as changes along positions:", len(rc(dP)), "bytes", flush=True)
print("rest:", rest_bytes(), flush=True)
judge("v0 baseline: plates v12 A=8 + tables c=34 (everything counted)", cfg_all("v12", 8), 34,
      note="round one's 30/30 point, re-judged on the whole machine with the undeveloped numbers counted", predicted="30")
