# Predictions, written before the whole-number Granite machine runs (2026-10-09)

The machine: `granite_base2.py`, the whole-number machine made the way `qwen/qwen_base2.py` was made. Its judge is
fp32 Granite-3.1-1B-A400M's own 16 greedy words on the 30 prompts (`refs_granite.json`).

1. **Words.** Qwen3-0.6B's whole-number machine gave 28 of 30 texts IDENTICAL (`qwen/check30.log`). Granite adds one
   place for a difference: the router's edge between the 8th and the 9th expert. At some positions fp32's own gap there
   is under 0.0001. Prediction: **20-27 of 30 IDENTICAL, about 24.** Every DIFFERENT text should split at or after a
   word whose fp32 lead is small (under about 0.15).
2. **Plates read.** For one position, each layer reads 8 of its 32 expert plates and leaves 24 dark. Over the 24 layers
   the expert plates are 91% of all the weights (1,208M of 1,334M). For one position, then, the machine reads about
   **32% of the weights** (attention 76M, router 0.8M, 8 lit experts 302M, top table 50M), and **68% stay dark**.
3. **Speed.** The run recomputes every position for each word, as `qwen_base2.py` does. It should take about 1-2 s a
   word, so roughly 10-20 minutes for 480 words.
