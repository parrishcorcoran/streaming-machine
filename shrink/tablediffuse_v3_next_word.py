"""TABLEDIFFUSE v3 -- carry each value's leftover to the SAME LINE OF THE NEXT WORD (the chain runs down each line
through all 50,257 words): in vocabulary order, and in a random word order (fixed seed 1).
The relation: the pile that cancels now is a line's sum over many words -- the light common to all words on a line,
which the top's race never feels (it shifts every word's score alike). Each word's own row still gets the difference
of two leftovers: grain power doubles for the entry and for the race between any two words that are not neighbours in
the chain.
Predicted: like v1 -- c=32 28-30, c=34 27-29, c=37 20-27; random word order the same within luck; bytes as v1.
    python3 shrink/tablediffuse_v3_next_word.py
"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td
td.run([dict(name=f"v3 carry to next word, vocabulary order, c={c}", c=c, how="words") for c in (32, 34, 37)]
       + [dict(name=f"v3 carry to next word, random word order seed=1, c={c}", c=c, how="words", word_seed=1)
          for c in (32, 34, 37)])
