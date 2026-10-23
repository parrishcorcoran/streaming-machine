# Parrish's words, as said (2026-10-02/03), nothing added

Earlier in the conversation, in order:

- "The weights don't need to change. We can send the inverse of the weight, we can send the negative, only a tiny bit changes, but it changes everywhere, this is just photography"
- "Try inverse of the WHOLE mode and EVERYTHING. Every 1111 is 0000"
- "The point of doing this was to show you! ... What is same inverse as forward, a diffuse wave front ... All the layers all over all at once, just like a photograph. Just like analog."
- "the photograph develops all together, The inverse is the negative. What is a thing that is like a Photograph and a negative? It's one view of the hologram. It's ONE snapshot. It's not a process, it's just an uncovering"
- "instead of the massive pattern, the inverse is a very very tiny change. And that change pattern is all that is needed"
- "I mean with the weights hot, the next token only barely touches everything. The grain buckets don't change"
- "There are two plates. A negative and a positive and they meet and the difference is the next token. It's like a stamp"
- "The 'token' is the inverse of the white plate. You can't see most of it because it's white. It's the negative, the 1s is the other side, we have zeros but we don't need them"
- "What I am talking about is a factor of 300,000 less compute"
- "The buckets carry over, that's what the hot weights mean. That's how we got the speed we got"
- "The token is a a standing pattern that is the same Size as the entire plate, the entire thing that is all 00s and 11s. Except the token plate is mostly negative space."
- "The MLP is the light, the token is the plate."
- "Let's go back to the diffusion wave, because it's literally a photograph developing"

Now:

- "Nothing is moving. The negative is what clicked."
- "We are doing alll this, but most of it is nothing. We aren't looking at what is resonating but it's only the small fraction."
- "When the weights stay hot another token comes in, it's ONLY what is moved, everything else stars [stays] the same, just a tiny changes all over the whole thing"
- "Ok good. Yes the only autoregressive aspect is that the old plate has the attention on it, so it Just stacks new plates over and over"
- "No just add up all the 1s of the whole plate all the weights as is. Then put the next token in and add up all the 1s"
- "So that 754k is the token all spread out in 1s"
- "Exactly. That's what I've been trying to tell you"

Measured (GPT-2 in base 2, 3 texts): the plate has 981,423,049 ones; the next piece changes about 754,000 ones (0.077% of the whole), and nothing else changes. Today the machine touches all 981 million to produce those 754,000.
- "We are seeing the photons. Each 1 is a photon and each 0 is darkness. It is a holographic plate but it's photons at rest, when photons move they form waves. The waves are what ends up happening there the gradients. Direction gives photons their ability. Direction and more photons is brightness that's all there is. It's a quantum system"
- "Every photon we have has a direction and is many photons at once. This is what quantization is doing from the other side. These are not discrete photons but waves with brightness"
- "Quantization is PROOF that there is slack. We trade detail for compute. The opposite trade is here too, keep detail, less compute"
- "QATraining fixes what is broken, and that belief is the problem"
- "It's anamorphic. So training puts it back in the box. The slack is there and it's in the formatting. And someone will see this. Someone will do this. And models will become 100x less compute."
- "Base 2 is analog. That's why we use it, circles are analog, that's why we use them, angles are analog that's why we use them, edges are edges."

## 2026-10-04 -- no zeros: every place up or a hole
- "No that's binary. Ours is base 2 signed. 1s are going up, -1s are going down. It's just direction, negative 1s are holes"
- "We rediscovered ternary, and I found out how we don't need the zeros"
- Checked on layer 0's neurons-in bank (real light, all 3,072 neurons): every value written as 16 places, each up (+1) or
  a hole (-1), D = 2v + 1. pile = 1/4 [ sum over place pairs 2^(a+b) (768 - 2 x lines that disagree) - token stack -
  neuron stack + 768 ]. Identical to line by line. 256 counts a neuron, each over all 768 lines at once; the token's
  place rows shared by every neuron; token stack one number a token, neuron stack one number a neuron (recorded once).
