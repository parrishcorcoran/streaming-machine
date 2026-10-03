# streaming-machine

GPT-2 (124M) with every number a whole number in base 2. No float exists while it runs.

Each new token adds only itself. Everything before it stands, with each layer keeping its keys and values. The new position flows through every layer and meets every weight once, spread over all the GPU's cores.

On a MacBook M4, 30 prompts × 16 words, greedy:

| | words/s | same words as fp32 GPT-2 |
|---|---|---|
| PyTorch, GPU, fp16 | 120 | reference |
| PyTorch, CPU, fp32 | 105 | reference |
| **streaming machine** | **180 including the prompt, 224 after it** | **29 of 30** |

The one difference is at word 4 of "It was a dark and stormy", where fp32's own lead is 0.0020.

## What it does

- **Weights:** 16-bit whole numbers, with one power of two per column. Layer norm's mean, gain and offset are folded into the weights at conversion, so what's left of it is the gauge.
- **Products:** made on the chip's integer multiplier. Each pile's products land in two narrow cells, the bottom 15 digits in one and the rest in the other, and the cells are read once at the end.
- **The gauge** (layer norm's spread): log2 is where the top 1 sits, plus a table read. A root is a halving and a divide is a negation.
- **The edge** (GELU): where the value arrives picks its share from a table.
- **The arrival** (softmax attention): the clock starts at the race's winner, and each score's distance from it reads the arrival table.
- **The word:** the race at the top. At each digit from the top, if anyone still in has a 1, everyone with a 0 drops out.

[SPEC.md](SPEC.md) has every number and operation.

## Run

```
pip install torch transformers numpy
python3 base2_convert.py      # GPT-2's fp32 weights -> whole numbers in base 2 (the only float in the project)
python3 base2_gpu.py          # laid out for the GPU
python3 make_refs.py          # fp32 GPT-2's own greedy words for 30 prompts
swiftc -O -o base2_stream base2_stream.swift && ./base2_stream all=1
```

`base2.py` is the same machine in Python, checked stage by stage against fp32 (`gauge=float`, `edge=float` and `arrival=float` switch a stage back).
