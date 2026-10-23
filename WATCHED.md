# Watched

## basics/code/stream_full.py — sounds wrong

GPT-2 with all 12 layers running every turn, each on what the layer below passed up the turn before.
Weights rounded onto the circles (24 bits, one gauge per column) but kept as floats; each layer is
GPT-2's own float code (GELU, attention, layer norm, multiplies). Values passed between layers go on a
dial, and an edge counts where a position changed. The top is read every turn.
Seen: the zeros climb one layer per turn, the top names junk until turn 13, then ' the' (= fp32),
and nothing moves from turn 14.
Not adds-only: the model runs as ordinary floats inside each layer.

## timing-substrate/flowing/flowsim.py — not helpful to watch

One MLP from GPT-2 (layer 5), one token, a clock counting 1–256. Each input value is turned into a
length of time it stays on (bigger value, longer on); every step, each input still on adds its weights
into the first layer. Batch: first layer finishes, GELU, then the second layer. Flowing: every step the
second layer takes the first layer's half-built totals through GELU and adds in only the change.
Seen: only final numbers. Both within a few percent of the real MLP output (flowing 2.5%, batch 3.6%).
Nothing to watch per step; only one MLP; GELU still a float function.

## timing-substrate/flowing/flowgpt.py — better, but incomplete

The whole of GPT-2, all 12 layers working every step on what the layer below passed up the step before.
The prompt arrives gradually: each input value grows step by step over 128 steps until it reaches its
full size. Inside each layer it is GPT-2's own float code (multiplies, GELU, exp, square root), rerun
whole every step on the partly-arrived input.
Seen: only the moments the top's answer changes: junk, ' the' at step 10, wobbles, holds from step 15
while the prompt is still arriving. Matches the normal run.
The gradual arrival is what stream_full was supposed to show. Not adds and delays inside the layers.

## timing-substrate/flowing/flowpulse.py — not helpful, incorrect

The whole of GPT-2, all 12 layers every step, prompt arriving gradually over 256 steps. Values going into
weight layers are sent only as changes: one up or down signal per step when the value has moved by one
small step; each signal adds that input's row of weights into totals that are kept, never cleared.
Layer norm, attention and GELU are still float functions every step.
Seen: only the end. Locks at step 56 on ' now'; the normal run says ' the'. Wrong answer.

## timing-substrate/pulsed/stream.py — under consideration

From before the change to edges. The whole of GPT-2, all 12 layers every step, on a clock of 4096 steps
(set with --ticks; easy to change the speed and the number of steps). Every number is a count of steps:
a value's size is how many steps it stays on, its sign which way it adds, and a value bigger than one
cycle goes round in laps. Weight layers only add. The prompt arrives gradually over the 4096 steps;
every step the whole model reruns on the part that has arrived and the top is read. GELU, layer norm
and attention are float functions (its own comment says so). 5 minutes on the Mac.
Seen (watched from outside; the script itself prints only the end): junk at first; then ' (', 'f',
' to'; ' France' and ' French' surface at about 25% arrived; ' the' at step 1251 (31% arrived) and it
never changes again. Matches fp32.

## timing-substrate/pulsed/flow.py — not helpful; wrong answer, cause not found

pulsed/stream.py with totals kept between steps: each value sends a signal only when it has moved by
one step size since it last sent (at most one signal per step), and only those weight rows are added.
Layer norm, attention and GELU still rerun in full each step as float functions. 4096 steps, 3 minutes
(about 11x fewer adds than stream.py, but only about 2x faster).
Prompt "The capital of France is", same as stream.py. Seen (watched from outside): ',', '-', ':',
' descriptions', ' description', ' 2018', then ' <<' from step 2575 to the end. Final answer ' <<';
the normal run says ' the'. Why it goes wrong was not figured out.

## timing-substrate/pulsed/gpuflow.py — companion to pulsed/stream.py; useful in its own way

The same machine as pulsed/stream.py (prompt arriving gradually over 4096 steps, every layer rerun
each step), but the 12 layers are grouped into one operation per stage and run together on the
graphics card. About 7 ms a step, 30 seconds a prompt instead of 5 minutes.
By default it stops at step 64 of 4096 and reports only MATCH/DIFFERS, so it shows 0/8 — the prompt has
barely arrived by then. Run the full 4096 steps (--max-ticks 4110), it matches: "The capital of France
is" -> ' the' at step 1249 (path through ' to', ' France', almost the same as stream.py); "Water boils
at a temperature of" -> ' about' at step 1189.
Shows that every layer working on the step before's output lets all layers run at once with no change
to the answer. The speed is the graphics card doing the same arithmetic (floats included), not adds
and delays. Not what we want, but helpful in its own way.

## tick-machine/dial_flow.py — intermediary stage

Not helpful on its own and might be confusing, but the steps are potentially heading in the right direction.
flowgpt.py's arrangement (all 12 layers working every step on what the layer below passed up the step
before; the prompt arriving gradually over 128 steps, set with --T) on the dial version of GPT-2 from
edge_model.py: each weight rounded to one of 65,536 positions (4096 x 16 laps, one gauge per column) and
then stored back as an ordinary decimal number; each neuron's output rounded to 4096 positions with laps;
the top read at 16,384. Inside each layer it is GPT-2's own float code. Weights are used whole every step;
nothing about a weight is spread over time.
Seen: per step, the strength of ' the' against the strongest other word. Junk until step 10, ' the' holds
from step 15, lead peaks +1.68 at step 25, eases to +0.3-0.6. Matches the normal run.
The "edges: 87 billion" line is calculated (weights x tokens x steps), not sent.

## tick-machine/dial_flow_order.py --T 16 --stream dial --input step

Prompt lands all at once; values passed between layers rounded to a dial (4096 + laps), a change counted
where a position moved. One layer goes quiet per step, 0 change from step 14, ' the' (= fp32) with the
same strength every step. Barely different from stream_full.py; no more helpful than it. Inside each
layer ordinary float arithmetic; the edges are counted after the fact, not sent.

## tick-machine/dial_flow_order.py --order shuffle — helpful: the abelian counter in action

Each step the 12 layers fire in a random order, each reading whatever is below it right now (including
what changed earlier in the same step). Prompt arrives over 128 steps. Same answer ' the', same final
strength, locks at step 7 instead of 15 (when the random order fires a layer then one above it, the
answer climbs several layers in one step). Firing order does not matter.

## tick-machine/dial_flow_order.py --init noise

Every layer's output starts filled with random junk instead of zeros; otherwise the defaults. Steps
1-12 show different junk; from step 13 every number is identical to the zero-start run, and it ends on
' the' (= fp32), locked at 15. The junk is replaced one layer per step and leaves no trace: the starting
state does not matter.

## tick-machine/dial_flow_order.py --perm 0 — useful

Layers reconnected in a random depth order [8, 0, 5, 9, 7, 4, 2, 10, 3, 1, 11, 6]; each layer unchanged,
only which feeds which. Junk for ~12 steps, then a run of speed words (' advancing', ' accelerating',
' clearing', ' sprint', ' advanced'), ending on ' fast' at step 109. The normal run says ' the'.
The wiring order matters, where firing order did not.

## tick-machine/dial_flow_order.py --gelu off | clip | relu

GELU replaced by nothing / a hard limit at +-3 / ReLU. All three settle on a wrong word: off -> ','
(path of word pieces: 'z', ' Fraud', 'oth', ' chemistry', 'acebook'), clip -> '-' (similar pieces),
relu -> ',' (junk close to the normal run's, never reaches ' the'). Parrish: probably only works
combined with one of the other settings.

## tick-machine/dial_flow_order.py --stream bits | dial1 | sign

What passes between layers. All three end on ' the' for "The capital of France is".
bits: one up/down bit per value per step; the receiver keeps the value and nudges it by a step that
grows on repeats and shrinks on reversals. Slow chase: lead falls to -10 by step 19, wanders, settles on
' the' at step 69. The only mode where a value is never sent, only nudged.
dial1: one 4096 dial with no laps (bigger values clipped). More wobble, locks at step 20.
sign: only + or - crosses. ' the' almost from step 1, locks at 10, then frozen at the same strength.
NOTES.md: sign matches on 1 of 5 prompts only; this prompt is that one.

## tick-machine/dial_flow_order.py --T 1 | 2 | 4 | 8 (prompt arriving as pulses)

Each input value is on for its width, rounded to a whole number of steps out of T, so T is also how
finely the prompt is written (T=1: each value all or nothing). Answers: T=1 ' of', T=2 ' (', T=4 ' the'
(= fp32), T=8 ' now'. Steps 1-12 identical in every run (the prompt has not reached the top yet).
Shorter runs (T + 13 steps), but the prompt itself is rounded and the answer changes; not a safe estimate.

## tick-machine/dial_flow_order.py — settings, and the summary for the whole script

Other prompts: "Once upon a time" ',' / "In 1492, Columbus" ' was' / "Water boils at" ' about' — all
match fp32. "Water boils at" with --order shuffle matches; with --stream sign it says ' the' (wrong).
--dial 4096/1024/256 and --act 65536/1024/256 all still give ' the' on the France prompt (one word, one
sturdy prompt; NOTES.md found coarse weights stop matching over longer generation).

SUMMARY: good for seeing the abelian aspects, and that's about it. One script, one switch each:
--order shuffle (firing order free), --init noise (starting state free), --perm (wiring order not free).
Worth remembering: --stream bits — only an up/down nudge crosses between layers and the receiving side
holds the value, and it still reaches the right word (slowly). The closest any of these come to "only
edges cross". Everything inside each layer is ordinary float arithmetic: not the adds-and-delays machine.

## tick-machine/dial_flow_feedback.py — worth looking into (a use for the streaming machine, not the machine itself)

Generation on the dial machine with every position in flight: each step a new position is added, and at
every position whatever the top currently names becomes the guess for the next, before anything settles.
32 positions, 200 steps max. Exact matches climb about one per 10 steps, ending 21 of 32; the first 17
words are the model's real text, the tail breaks into repeats. Only exact matches were checked — the
measure that matters is whether each word is in the model's top 5 (a near-tie is not wrong); not yet checked.
Known idea: Jacobi decoding (Santilli et al. 2023), lookahead decoding (Fu et al. 2024) — pretrained
models, no training; real gains come from guessing 2-4 positions ahead and checking. What is specific
here: a guessed word starts climbing the layers while the one before it is still settling.
Worth looking into: few positions ahead, judged by top-5.

## tick-machine/qwen_stream.py — portability

Purpose (Sep 19): after qwen_circle.py put Qwen3-0.6B on the GPT-2 circle recipe, this ran it on the
streaming arrangement to show it carries over to a differently built model, and drew the picture
~/tick-machine-writeups/light_cone_qwen3.svg (from qwen_cone.npz, which it writes). Never written up in
MODELS.md.
What it is: all 28 layers working every step on the step before's output; whole prompt lands at once.
Weights and values between layers whole numbers on dials (dial sizes set by measuring the float model on
three prompts first); weight layers multiply; layer norm, rotation, attention, SiLU are floats. Not adds
and delays inside a layer.
Seen: top exactly zero ('!') for 27 steps, ' Paris' (= fp32) at step 28, lead +1.85 vs fp32 +3.16.
Each step exactly one layer changes; each layer changes exactly once in the whole run, then holds.
Qwen has no bias terms, so layers above the prompt stay perfectly silent (GPT-2's chatter).
Probably not very useful on its own. Possibly useful: only 28 of 812 layer-steps ever have work to do.

## basics/code/stream13.py — the analog view of stream_full.py

Plain float GPT-2 (no rounding), all 12 layers working every step on the step before's output, whole
prompt at once. Each step prints one number per layer: the 768 values at the last token summed into one
level, written as angle:laps on a 4096 dial — the reading one probe on one wire would give in analog.
Each layer's level jumps around until the wave reaches it, then locks and holds, layer by layer up the
diagonal; ' the' (= fp32) from step 13. The summed level cannot be reversed back into the 768 values.
Same machine and same words as stream_full.py; not faster — the sum is computed only for display.

## basics/code/stream_sentence.py — potential uses

GPT-2 with all 12 layers working every step on what the layer below passed up the step before; GPT-2's own
float code inside each layer. Nothing is ever reset: when everything has gone still, the top word is added
as a new token at the bottom. 24 words, 14 steps each, text identical to the normal run. Earlier tokens stay
settled; each new word is a small wave (one token's 768 values per layer) climbing past them.
Words 2 and 4 show the old last token shifting 1-5 dial steps and returning: float32 rounding when the token
count goes odd -> even, not the model.
The settled state depends only on what has been put in, so the next word can start before the current one
settles, and a wrong guess can be replaced: always correct, misses cost only time.
Others built on this, to watch: dial_pipe.py, dial_hedge.py, dial_branch.py (tick-machine); bound.py,
pipeline_spec.py (basics, Mac only).

## basics/code/stream_watch.py — helpful; the inside-a-layer view

One real GPT-2 MLP (layer 5), one token, split between two sides that never share weights: B holds the first
bank, A the second (whole numbers up to 32,767; they never move). Each of 768 input values crosses as one edge;
in B each pours into a pile for as many steps as its weight's size (the product is time); at lap end each pile
goes through GELU and sends one edge back; A pours the same way. You watch four piles fill over the lap; the
pattern is 0.83 of its final shape after 64 of 32,767 steps. Random order in two halves gives identical piles.
Matches the real MLP to ~0.5%. Only 3,840 edges cross per token; 4.7M weights stay put.
Limits: one MLP, one token (no attention, no layer norm); GELU still a float function; the pour is computed
at chosen moments, not stepped; the last check (phases instead of pours) is a negative tacked on the end.
Building block: stream_sentence.py's machine from outside, this view inside each layer.

## basics/code/stream_neuron.py — important; the inverse side of bandwidth

Written for the analog work. One token (" France") into one neuron of GPT-2's first layer, weights and
inputs on two circles of 4096 (16.7M positions). Each weight is a starting time: a weight of size p pours
for the last p steps of a 4096 x 4096-step run (bigger starts earlier); each input value is how much pours
per step; the pile is everything poured. No multiplying. Ends at +0.40550; GPT-2 says +0.40550 — exact.
All 768 pour at once, so one neuron or all 3,072 take the same run (16.7M steps, 6.7 s at 400 ns a step):
the cost is time, not how many weights there are — the inverse side of bandwidth.
Levels are computed at chosen moments, not stepped one by one.
The answer's timing depends on how weights are timed: here most weights start late, so the pile moves mostly
at the very end and swings sign mid-run; in stream_watch.py they start at once and the shape is 83% there
after 64 steps. 400 ns a step is an assumed rate, not measured.

## basics/code/stream_collapsed.py — useful; companion to stream_neuron.py

The same pour as stream_neuron.py for the whole bank of 2,304 neurons (one token, GPT-2's first layer). Each
weight's size split into whole laps plus a remainder within one lap: the whole laps add the same amount every
lap, so they are added at the very start in one go; only the remainder needs time, and it fits in one lap of
4,096 steps. Neuron 1: -0.07441 at once + -0.00043 over the lap = -0.07484, as before; all 2,304 neurons match
GPT-2 to 6e-7. One lap instead of 4,096 (1.6 ms instead of 6.7 s at an assumed 400 ns a step).
Of a value's two circles, only the position within the lap costs time. The "laps at once" step is a multiply
in the code, standing in for the pour — fine, the adds are a demonstration anyway.

## basics/code/streaming_ticks.py — the whole-model version of binary_ticks.py

stream_full.py's machine with the four weight banks of every layer done in 24 binary steps: each weight split
into its 24 binary digits; at step j the inputs whose weight has digit j set are added in and the pile doubles.
GELU, layer norm, attention are GPT-2's own float code. Same path as stream_full.py; ' the' (= fp32) from step 12.
Counts 58.6 billion adds and 129 million doublings for one word. Slow: 13 minutes (Python whole numbers, one token
at a time, 24 passes per bank). Its "multiply 0" is not measured: that counter is never touched.
Point: the time line — a weight's wait is 16.7M steps (stream_neuron.py), 4,096 with the laps at once
(stream_collapsed.py), 24 per bank in binary digits (binary_ticks.py) — and the whole streaming model still works.
To watch: basics/code/binary_ticks.py (one bank, the 24-step idea directly), and radix_sweep.py beside it.

## basics/code/settle_early.py — not helpful, incorrect

Plain float GPT-2, all 12 layers working every step, prompt in place at step 1. Prints the top's word, its lead,
and "still moving" (the total change of every value, added into one number): rises to ~235,000 by step 4, falls
to 0 at step 13. ' the' at step 10, ',' at 11, ' the' from 12. Taking the answer at a fixed earlier step over 8
words: step 4/6/8/10 give one word repeated (1/8, 1/8, 0/8, 2/8); step 12 gives 8/8. Before step 12 the top is
answering from the empty layers above the prompt, so "knew 3 turns early" is not knowledge; step 12 = 12 layers.
Leads into bound.py (take it when it can no longer change) and pipeline_spec.py.

## basics/code/bound.py — not helpful, incorrect

Eight rules for "take the word when it can no longer change", tried on 3 prompts x 8 words. The rules that work
(still-moving below 10% of peak, or below 25% with a lead) take the word at about step 12, when the prompt reaches
the top: under one step saved out of 13. Lead-only rules fail at step 1 (0/24). Not bounds that are always right:
thresholds tried on 3 prompts and kept if they happened to work.

## basics/code/pipeline_spec.py — not helpful, incorrect

Says the next word starts while the current one settles, with take-backs; the guess-ahead code is just `pass`,
so 0 guesses, 0 take-backs. Actually takes each word when still-moving drops below 10% of peak: 12 steps instead
of 13 (1.08x), text identical. The real early-guessing family is dial_pipe.py, dial_hedge.py, dial_branch.py.

## holographicplate/stream.py — extremely useful, but limited by speed

The closest yet to the whole model done the analog way. Qwen3-0.6B on whole numbers, one pulse per token, up to
28 in flight. Every product a pour (one shifted add per binary digit of the weight); attention as arrivals (a
table of halvings, then one divide per pile); the position turn by CORDIC shift-and-add; the bend (SiLU) as the
lap's edge — a table lookup then pours, NOT a float function; the exit poured through all 151,936 word rows.
Says 'Paris.'. Still multiplies (whole numbers): layer norm's squares and gain, the turn's gain correction; one
float step putting attention scores onto the halving table.
Having the bend as adds and table lookups is one of the things needed: every gate the model has, done analog-like.
Limited by speed: ~6 s a step, 9 minutes for 2 words. Mostly waste — every use of a weight layer re-derives the
binary digits of the whole matrix (including the 151,936-word table) from scratch; building them once would
roughly halve it, the graphics card more. The rest is the real cost of doing 22 passes instead of one multiply.
Display shows little: only the schedule and the exit word; its "edges fired" column always reads 1.00.

## babylight/stream_onebit.py — a sweep of quantization; not useful for streaming

stream_sentence.py's machine with every weight in the four banks cut to N bits (default 1: sign x column
average). GELU, layer norm, attention are GPT-2's float code. The machine behaves the same (14 steps a word).
Text: 1 bit '------', 2 '......', 3 ' the place.', 4 ' the capital of the United States', 6 and 8 bits
identical to fp32. The early ' the' at step 3 in the 1-bit run is not the prompt: steps 1-12 are identical for
every prompt (the empty layers above the prompt). Possibly remembered instead: holographicplate PICK_UP_HERE —
Qwen's word table (the plate) at one bit stays fluent.
