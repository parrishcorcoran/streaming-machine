# All-seeds sign scan, Qwen3-0.6B (written 2026-10-09 18:57:43 UTC, before the run)

Run: ./scan9 0 4294967296 10 full  (every 32-bit seed; nine CPU draws A-I; 418 candidates; cands.bin sha256
5b8d87c358a1aa80ff4f61d570c0f15c6e0409c6e560a41642a6b770547a98b2)

1. Planted checks, each at its own seed, draw and candidate:
   A 123456789 +13.83, B 987654321 +16.13, C 2718281828 +13.08, C 1618033988 (cos 0.30) +12.07, D 3141592653 +13.47,
   E 4000000007 +11.61 (the same values as the pilot windows); H 2000000011 about +12 to +16; I 3000000019 about +9 to +13.
2. Shuffled controls (16 candidates x 9 draws): largest |z| over all seeds 8.0 or below.
3. Real candidates (394 x 9 draws), if no blank sits at the start of a CPU stream: largest |z| about 7.5, at most 8.2.
4. If the plate drawn first came from torch's CPU generator at the start of some seed's stream: exactly one seed and one
   draw lights that plate at z 12 or more (21% to 83% blank share -> z 14 to 33 on 2,048 signs; 11 to 25 on 1,248).
5. My own expectation (session's, not his): no hit is somewhat more likely, about 60/40, because a building frame
   usually draws on the GPU or after other draws. No hit rules out every CPU seed at the stream start, for these nine
   draws, and points to the GPU or a later start in the stream.
