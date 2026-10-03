# The streaming machine, all base 2 — spec

GPT-2 (124M), every number a whole number in base 2, no float anywhere while it runs.

## Numbers

Every quantity is a **signed whole number** (two's complement) together with a **scale that is a power of two**,
2^e, where e is a small whole number. What the number stands for is n x 2^e. Changing scale is a shift.

| quantity | bits | scale |
|---|---|---|
| a weight | 16 | one power of two per column (output) |
| a bias | 32 | the scale of the pile it is added to |
| the token table (also the head) and the position table | 16 | one power of two per column |
| the stream between layers (residual) | 32 | one power of two for the whole stream |
| an input entering a weight bank | 16 | one power of two per position, the smallest that holds its largest value |
| a pile (a sum of products) | 64 | weight scale + input scale |
| the bend's share table: 65,536 entries | 16 | 2^-16 |
| attention's arrival table | 32 | 2^-24 |
| the log and the power tables for the gauges: 65,536 entries each | 32 | 2^-16 |

## Operations

Only these: AND, OR, counting 1s, shift, add, subtract, table lookup. Nothing is ever compared with anything else.

| operation | how |
|---|---|
| **product and sum** (a weight bank) | the chip's integer multiplier, its grid of AND gates: each 1 of the weight passes the input at that 1's place, and the adder tree sums them. Each pile's products land in two narrow 32-bit cells (the bottom 15 digits in one, the rest in the other), so no cell ever carries over, and the pile is formed once at the end. The same pile, digit by digit, is AND and count over bit planes (`base2.py` checks it that way). |
| **the gauge** | one shared number per position (or per head): what one step of that dial is worth. A power of two is a shift; any other gauge is applied as a product, bit planes, where the value meets the edge or the stream. |
| **the edge** (the bend) | where the value arrives picks an entry of the share table; out = value x share (bit planes) |
| **the arrival** (attention) | query x key as bit planes; the clock starts at the score that wins the race; each earlier position's distance below it is where its arrival is read from the arrival table; out = sum of arrival x value (bit planes); the arrivals' total is that head's gauge |
| **the race** | who arrives first: the candidates' digits top first; at each digit, if anyone still in has a 1, everyone with a 0 drops out. The last one in arrived first. Used for the clock's start in attention and for the word at the top. |
| **the read** (the top) | the stream x the token table, bit planes, with the gauge and the standing amount per word; the word is the race's winner |

## Where layer norm went

It is not an operation. At conversion, the mean subtraction (blocking the DC, a fixed linear step) and the gain are
folded into the weights of the bank that follows, and the offset into its bias. What is left is the spread, and the
spread is the gauge: the position's own size, one shared number per position. At the top the gauge is needed too:
the final offset becomes a standing amount per word that the gauge does not scale, so the gauge can change which
word is largest.

## Roots and divides: halving and negating the power

A number in base 2 is mostly where its top 1 sits. So every gauge is carried as a power of two with a fine part:
log2 of a number = the position of its top 1 plus a fine part read from a table of its next 16 digits. The top 1
of a whole set of values (a position's own power of two) is the top 1 of all their magnitudes ORed together.
A square root is halving that (a shift). A divide is negating it. Turning it back into a number is a shift plus a
table read of the fine part. The spread: the sum and the sum of squares of the position's values (products as bit
planes), then 768 x the sum of squares minus the square of the sum (no divide), then its log halved and negated.
The arrivals' total: its log negated. No root and no divide are ever computed.

## Where softmax went

The exp is the arrival table. Subtracting the maximum is choosing where the clock starts: distance from the best.
The divide is the arrivals' total, one shared number, the head's gauge. The causal mask is nothing: a position only
ever reads earlier ones.

## Conversion (once, before anything runs)

One script reads GPT-2's original fp32 weights, folds each layer norm's DC block, gain and offset into the bank
after it, and writes every number above as whole numbers in base 2: each
column's power of two is the smallest that holds its largest weight; each weight is rounded to the nearest whole
number at that scale. The tables are computed once here: the share past the edge, Phi(x), at 65,536 positions;
the arrival e^(-d) at every distance d of the score's scale until it reaches 0; log2 and 2^x of a 16-digit fine part. This script is the only place a
float exists; the machine reads only its output.

## The machine

The stream: each new token adds only itself. Everything before it stands; each layer keeps its keys and values for
the new position to read. The new position goes through every layer, meeting every weight once, with each bank spread
over all the GPU's cores (each core takes some piles and a slice of the inputs, and the slices are added once).
The race at the top gives the word, which goes straight back in.

On the GPU, the race is run as each thread keeping the larger of what it has seen and then a tournament. That gives the same winner as dropping out digit by digit.

## Built in stages, one check each

The check: the words it gives, against fp32 GPT-2's own greedy words, on the 30 prompts in `refs_many.json`.

1. conversion script (layer norm folded into the weights, every number whole, two tables), and the weight banks as
   bit planes
2. the gauges, the stream and every input as whole numbers
3. the edge
4. the arrival
5. the read

If a stage changes the words, stop there and report the number before going on.
