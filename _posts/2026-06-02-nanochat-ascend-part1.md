---
title: "Nanochat-Ascend: Training Karpathy's Nanochat on Ascend NPU (Part 1)"
layout: post
description: "A walkthrough of training Karpathy's nanochat on Ascend NPUs, covering tokenization, GPT pretraining, SFT, and RL."
use_math: true
---

## 0. Introduction

I have worked on LLM inference for two years, and I have always been curious about how an LLM is trained to reach such a human-like level of intelligence. I knew I would not deeply understand it unless I trained one from scratch myself. So I started the [nanochat-ascend](https://github.com/leideng/nanochat-ascend) project during the Lunar New Year holiday and spent about three months on it in my spare time.

I started from [Andrej Karpathy](https://karpathy.ai/)’s famous [nanochat](https://github.com/karpathy/nanochat) project. Karpathy trained it on 8×H100 GPUs; I only *slightly* adapted the code so I could train it on an Ascend A3 machine (8×910C NPUs, or equivalently 16×910B NPUs).

Adapting Karpathy’s `nanochat` code to Ascend NPUs turned out to be straightforward. That highlights how portable his implementation is, even on a platform he never tested. It also suggests that the Ascend ecosystem—especially its PyTorch support—is already relatively mature.

Training `nanochat-ascend` was straightforward, but understanding how it works took most of my time. I wrote this blog to summarize both the theory and the code path for training a GPT-like LLM.

Before diving into the details, I want to share a few lessons I took away from the project.

> **Note**
>
> LLMs are probabilistic because of next-token prediction, so their outputs are unreliable by nature. It would be risky to treat them as ground truth and build everything on top of them without verification. Reliable tools, agents, and human judgment are essential if we want LLMs to actually boost productivity. I do not think AI will replace people outright, but it does raise the bar for what we need to know.

The sections below walk through **nanochat-ascend** training (based on d20). Links to the code, data, tokenizer, and models:
- Code: [https://github.com/leideng/nanochat-ascend](https://github.com/leideng/nanochat-ascend)
- Dataset: [https://huggingface.co/datasets/leideng/nanochat-ascend-dataset](https://huggingface.co/datasets/leideng/nanochat-ascend-dataset)
- Tokenizer: [https://huggingface.co/leideng/nanochat-ascend-tokenizer](https://huggingface.co/leideng/nanochat-ascend-tokenizer)
- Pretrain/Base Model: [https://huggingface.co/leideng/nanochat-ascend-d20-pt](https://huggingface.co/leideng/nanochat-ascend-d20-pt)
- SFT Model: [https://huggingface.co/leideng/nanochat-ascend-d20-sft-pt](https://huggingface.co/leideng/nanochat-ascend-d20-sft-pt)
- RL Model: [https://huggingface.co/leideng/nanochat-ascend-d20-rl-pt](https://huggingface.co/leideng/nanochat-ascend-d20-rl-pt)

## 1. Tokenization

### 1.1. What is Tokenization?

LLMs process text, but computers operate on numbers. Tokenization bridges this gap by converting raw text strings into discrete token IDs that a model can learn from. The core design tradeoff is **vocabulary size $V$ vs. sequence length $L$**.

There are three common types of tokenization:
- **Character-level tokenization**: Split text into individual characters (or bytes). This keeps the vocabulary tiny (for bytes, at most 256 symbols), but sequences become very long resulting in high computation cost.
- **Word-level tokenization**: Split text by words. Sequences are short, but the vocabulary becomes very large and open-ended. Unseen word and misspellings become hard to handle.
- **Subword-level tokenization** (BPE / WordPiece / Unigram): Split text into frequently used chunks (smaller than words but larger than individual characters). This balances the two extremes such that vocabulary is manageable, sequence length is moderate, and any text can still be represented by decomposing into smaller pieces. Though it often breaks words into chunks that lack consistent semantic meaning, it is the de facto standard in LLMs.

Below is a concrete comparison to tokenization the same sentence: `Training nanochat on Ascend is extremely fun`

| Type | Example Split | Typical Vocab Size $V$ | Sequence Length $L$ | Main Strengths | Main Weaknesses |
|---|---|---|---|---|---|
| Character-level | <code>T&#124;r&#124;a&#124;i&#124;n&#124;i&#124;n&#124;g&#124;<br>&#124;n&#124;a&#124;n&#124;o&#124;c&#124;h&#124;a&#124;t&#124;<br>&#124;o&#124;n&#124; &#124;A&#124;s&#124;c&#124;e&#124;n&#124;d&#124;<br>&#124;i&#124;s&#124; &#124;e&#124;x&#124;t&#124;r&#124;e&#124;m&#124;e&#124;l&#124;y&#124;<br>&#124;f&#124;u&#124;n</code> | ~100 to 300 <br> (or 256 bytes) | 44 (Longest) | Full coverage, <br> robust to typos/unseen words | Very long context, <br> high FLOPs |
| Subword-level | <code>Tr&#124;ai&#124;ning&#124; nan&#124;o&#124;chat&#124;<br>on&#124;Asc&#124;end&#124; is&#124;<br>ex&#124;tre&#124;me&#124;ly&#124; fun</code> | 8K to 100K <br> (often 32K to 50K) | 15 (Medium) | Best practical balance of <br> coverage and efficiency | Fragmented semantics |
| Word-level | <code>Training&#124; &#124;nanochat&#124;<br>&#124;on&#124; &#124;Ascend&#124;<br>&#124;is&#124; &#124;extremely&#124; &#124;fun</code> | 100K to millions | 13 (Shortest) | Simple and short sequences, <br> align with human semantics | Huge embedding table, <br> cannot handle Out-of-Vocabulary (OOV) words |


From our later GPT-like architecture perspective, tokenization directly affects the computation and memory cost. In particular, if we increase the vocabulary size $V$, we can reduce the sequence length $L$ for a given language string, and thus

- **Reduce FLOPs per forward\&backward pass**: shorter token sequences reduce attention/FFN computations.
- **Reduce KV-cache memory**: each token adds one key/value pair per layer, so fewer tokens save memory and improve throughput.
- **Reduce learning difficulty**: larger vocabulary size often brings better token boundaries and makes next-token prediction more robust.
- **Increase information density**: Few tokens to represent the same language string increases the information density.

However, large vocabulary size $V$ also brings a problem,

- **Increase the embedding table and LM head (unembedding) matrix**: large vocabulary size results in large embedding table and large LM head matrix, both of which have {::nomarkdown}$V \times d_{\text{model}}${:/nomarkdown} elements.

So tokenization is essentially a learned compression scheme that balances computational efficiency and representational completeness before the model ever sees the data. However, in practice, we do not jointly train the tokenizer and the GPT network. We simply train the tokenizer first and then apply it to train GPT network.

### 1.2. Theory of Tokenization

Now I will formulate the tokenization problem theoretically. First of all, let us define a language string by a byte string.

Denote the set of all byte values as

$$
\Sigma := \left \lbrace b_{0}, b_{1}, \cdots, b_{255}\right \rbrace.
$$

Here for notation simplicity, we use {::nomarkdown}$b_{i}${:/nomarkdown} to mean the byte to represent decimal number $i$. For example,
{::nomarkdown}$b_{97}${:/nomarkdown} is the byte representation of decimal number 97 or hexadecimal number 0x61, which is thus char 'a'. In python, it outputs like this

```python
x=bytes([0x61])
y=bytes('a', 'utf-8')
z=bytes([97])
print(f"type(x)={type(x)}, x={x}")
print(f"type(y)={type(y)}, y={y}")
print(f"type(z)={type(z)}, z={z}")

#output
type(x)=<class 'bytes'>, x=b'a'
type(y)=<class 'bytes'>, y=b'a'
type(z)=<class 'bytes'>, z=b'a'
```

A byte string of length $n$, called $\vec{s}$ is a finite sequence:

$$
\vec{s} := (s_1, s_2, \cdots, s_n) \in \Sigma^n, \quad s_i \in \Sigma, \forall i
$$

The set of all byte strings is then the Kleene star:

$$
\Sigma^{\ast} := \bigcup_{n=0}^{\infty} \Sigma^n
$$

where $\Sigma^0 = \lbrace\varepsilon\rbrace$ contains only the empty string.


We can formulate the tokenization problem as a **lossless compression** problem. Fix a finite vocabulary $\mathcal{V}$ with $\lvert\mathcal{V}\rvert = V$ (for example token IDs $\lbrace 0,1,\cdots,V-1 \rbrace$). A byte string $\vec{x} \in \Sigma^{\ast}$ is the raw text we must recover exactly.

We seek an **encoder** and a **decoder**

$$
T_{enc} : \Sigma^{\ast} \to \mathcal{V}^{\ast}, \qquad T_{dec} : \mathcal{V}^{\ast} \to \Sigma^{\ast},
$$

such that reconstruction is exact,

$$
T_{dec}(T_{enc}( \vec{x} )) = \vec{x} \quad  \forall \vec{x} \in \Sigma^{\ast},
$$

and the encoded sequence {::nomarkdown}$T_{enc}(\vec{x}) = (t_1,\cdots,t_{L(\vec{x})})${:/nomarkdown} has length $L(\vec{x})$ in tokens. The goal is to **minimize the average number of tokens** under a data distribution {::nomarkdown}$P_{\text{data}}${:/nomarkdown} over byte strings (or over strings seen in training):

$$
\begin{aligned}
&\min_{T_{enc},T_{dec},\mathcal{V}} \mathbb{E}_{\vec{x} \sim P_{\text{data}}}\big[L(\vec{x})\big] \\
&\quad \text{subject to} \quad |\mathcal{V}| = V, \quad
T_{dec}(T_{enc}(\vec{x})) = \vec{x}, \quad \forall \vec{x} \in \Sigma^{\ast}.
\end{aligned}
$$

Among all such pairs {::nomarkdown}$(T_{enc},T_{dec})${:/nomarkdown} with fixed vocabulary size $V$, we want the shortest typical codeword length.
For finite corpora, natural decision variants of this compression-style tokenization problem are **NP-complete** [[1]](https://aclanthology.org/2025.acl-long.1365.pdf).
Thus, finding a globally optimal {::nomarkdown}$(T_{enc},T_{dec},\mathcal{V})${:/nomarkdown} is **computationally intractable** in the worst case. Practical tokenizers (BPE, Unigram, etc.) approximate this objective by **merging frequent substrings into single vocabulary items**, which shortens $L(\vec{x})$ while preserving lossless decode back to bytes. This is the most fundamental principle in data compression.


### 1.3. Byte-Pair Encoding (BPE) Algorithm

**Byte-Pair Encoding (BPE)** is the subword tokenizer used in ChatGPT and it becomes the de facto standard in LLMs. It was introduced for compression by Gage (1994) [[3]](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM) and adapted for NLP by Sennrich et al. (2016) [[4]](https://aclanthology.org/P16-1162/). Starting from the byte alphabet $\Sigma$, BPE greedily **merges the most frequent adjacent symbol pairs** until the vocabulary reaches size $V$.

The key data structure for BPE algorithm is the `mergeable_ranks` (I call it **merge list** though it is a dictionary), which is a dictionary from byte string (token string) to integer (rank, which is also token id). The ranks correspond to merge priorities. We can see OpenAI's educational implementation (slow but clear) in [`tiktoken/_educational.py`](https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py).

We can also see the merge list of `cl100k_base` trained by OpenAI,

```python
import tiktoken
cl100k_base = tiktoken.get_encoding("cl100k_base")
mergeable_ranks=cl100k_base._mergeable_ranks
# print dict
print(mergeable_ranks)
# output
{b'!': 0, b'"': 1, b'#': 2, b'$': 3, ... , b'\xad': 255, b'  ': 256, b'    ': 257, b'in': 258, ...,  b' Conveyor': 100255}
```

#### Training (building the merge list)

1. **Initialization.** Every byte $b \in \Sigma$ is a token. Initialize the merge list as `merge_list[b_i]=i, for i=0,1,...,255`.
The corpus is a list of words where each word is a byte string.
Note that the space often "belongs to" the next word, not the previous one, and
is often represented by `Ġ`.
2. **Counting.** Count adjacent token pairs $(u,v)$ in the corpus. Note that an `aaa` block contributes two `aa`, not one.
3. **Merging.** Let $(u^\star, v^\star)$ be the most frequent pair (ties broken arbitrarily). Append the merge $\mu^{\star} = (u^\star, v^\star)$ as a new byte string to the **merge list** whose rank is the next rank of the existing merge list, namely, `merge_list[\mu^*]=len(merge_list)`. Replace every **non-overlapping, left-to-right** occurrence of $u^\star v^\star$ in the corpus with that new symbol. The new corpus will be used for iteration.
4. **Termination.** Stop after $V - \lvert\Sigma\rvert$ merges (or when no pair has count $\ge 2$).

The merge list is the serialized output of training: it records **in which order** pairs were fused, and it will be used during encoding! The vocabulary is $\Sigma$ plus the byte strings yielded by each merge, which is simply the set of all keys in the merge list.

#### Merge list governs both encoding and decoding

As I mentioned before, the entire BPE codec is specified by the dictionary:

- **`merge_list`**: `dict[bytes, int]` mapping each token’s byte string to its rank (which is also its token ID). Keys are exactly the vocabulary — every byte in $\Sigma$ at init, plus one new key per training merge. There is no separate vocabulary structure.

**Decoding** is straightforward: given token IDs {::nomarkdown}$(t_1,\ldots,t_L)${:/nomarkdown}, look up the byte string for each ID (the key $s$ with `merge_list[s] == t_i`, or an inverse map built once from the dict) and concatenate:

$$
T_{dec}(t_1,\ldots,t_L) = s_1 \,\|\, \cdots \,\|\, s_L.
$$

Concatenation is lossless because every token expands to bytes.

**Encoding** repeatedly merges adjacent symbols whose concatenation is already a key in `merge_list`. At each step, among all applicable pairs in the current sequence, merge the one whose merged byte string has the **smallest rank** (highest priority); scan left-to-right within each pass. Repeat until no key in `merge_list` matches any adjacent pair. Different rank orderings can yield different segmentations for the same string, so `merge_list` is part of the codec specification.

#### Worked example: training, encoding, and decoding

Train on corpus $\lbrace aa, aab, aab \rbrace$ with three words (bytes `a` = `0x61` = 97, `b` = `0x62` = 98).

| Step | Most frequent pair | `merge_list` (relevant entries) | Corpus after merge |
|------|-------------------|--------------------------------|--------------------|
| 0 | — | `{..., b'a': 97, b'b': 98, ...}` (all 256 bytes) | $aa,aab,aab$ |
| 1 | $(a,a) × 4$ | `{..., b'a': 97, b'b': 98, ..., b'aa': 256}` | $X,Xb,Xb$ |
| 2 | $(X,b) × 2$ | `{..., b'a': 97, b'b': 98, ..., b'aa': 256, b'aab': 257}` | $X,YY$ |

Here $X$ denotes `b'aa'` (rank 256) and $Y$ denotes `b'aab'` (rank 257).

**Encode** `aab`:

1. Start: `b'a'` \| `b'a'` \| `b'b'`
2. Merge `b'aa'` (rank 256) on the first two symbols: `b'aa'` \| `b'b'`
3. Merge `b'aab'` (rank 257): `b'aab'`
4. Token sequence: `[257]`.

**Encode** `aaaa`:

1. Start: `b'a'` \| `b'a'` \| `b'a'` \| `b'a'`
2. Merge `b'aa'` (rank 256) at positions 1–2: `b'aa'` \| `b'a'` \| `b'a'`
3. Merge `b'aa'` on the remaining `b'a'` \| `b'a'`: `b'aa'` \| `b'aa'`
4. No rank-257 rule applies (`b'aab'` is not a substring). Token sequence: `[256, 256]`.

**Decode** `[257]` → `b'aab'`. **Decode** `[256, 256]` → `b'aa'` \| `b'aa'` = `aaaa`.


#### BPE is not a prefix-free code

In **prefix-free** (instantaneous) codes such as Huffman coding, no codeword in the vocabulary is a prefix of another. Decoding a bitstream is unambiguous: read bits until they match exactly one symbol, emit it, continue.

However, one key in `merge_list` **can** be prefixes of another different key. After step 1 above, the dict contains both `b'a'` and `b'aa'`. The byte string `b'a'` is a **prefix** of `b'aa'` in the vocabulary. If you tried to segment raw bytes `aab` **without** merge ranks, both `[a][a][b]` and `[aa][b]` (and, after step 2, `[aab]`) are consistent with the keys alone.

We can compare BPE with Huffman coding as follows.

| Property | Huffman Coding (compression) | BPE (tokenizer) |
|----------|----------------------|-----------------|
| Codeword prefix-free? | Yes (by construction) | **No** — `b'a'` vs `b'aa'` |
| Ambiguity removed by | Code tree | **`merge_list`** + greedy encode rule |
| Decode from stream | Unique bit parsing | inverse map of **`merge_list`** |

So BPE is a **lossless segmentation codec** whose “codewords” are variable-length byte strings with prefix overlaps; `merge_list` plays the role Huffman’s prefix property plays for bitstreams.

#### Performance guarantee 

Paper [[2]](https://aclanthology.org/2023.findings-acl.38v2.pdf) formalizes BPE training as maximizing **compression utility** on a string $\vec{x}$:

$$
\kappa_{\vec{x}}(\mu) := |\vec{x}| - \big|\mathrm{APPLY}_\mu(\vec{x})\big|,
$$

where {::nomarkdown}$\mathrm{APPLY}_\mu(\vec{x})${:/nomarkdown} is the symbol sequence after applying merge sequence $\mu$ left-to-right, and $\mu$ must be a **valid** merge sequence (each merge only combines symbols that already exist at that point). The training problem is

$$
\mu^\star := \arg\max_{\mu \in \mathcal{M}_\Sigma,\ |\mu| = M} \kappa_{\vec{x}}(\mu).
$$

Greedy BPE (pick the most frequent pair, merge, repeat) outputs $\mu^\dagger$. Under **hierarchical sequence submodularity** of {::nomarkdown}$\kappa_x${:/nomarkdown}, Zouhar et al. prove

$$
\frac{\kappa_x(\mu^\dagger)}{\kappa_x(\mu^\star)} \;\ge\; \frac{1}{\sigma(\mu^\star)}\Bigl(1 - e^{-\sigma(\mu^\star)}\Bigr),
$$

where $\sigma(\mu^\star)$ is the **total backward curvature** of an optimal merge sequence. Empirically, $\sigma(\mu^\star) \approx 2.5$ on natural-language corpora, giving a lower bound of roughly **0.37** on the ratio of compression achieved by greedy BPE versus the optimal $M$-merge sequence on that string.

**What this means in practice:**

- Greedy BPE is not guaranteed globally optimal (unlike Huffman on a fixed symbol set with known frequencies), but it has a **provable approximation ratio** based on the submodularity on merge gains.
- The guarantee is about **training-time compression** (how many symbols remain after $M$ merges), not about downstream LM perplexity—but it justifies BPE as a principled approximation to the intractable tokenization objective.


### 1.4. Train and Evaluate the Tokenizer

Karpathy’s nanochat implements GPT-4-style BPE in two Python files I did not modify: [`nanochat/tokenizer.py`](https://github.com/leideng/nanochat-ascend/blob/main/nanochat/tokenizer.py) (`RustBPETokenizer` — [rustbpe](https://github.com/karpathy/rustbpe) for training, [tiktoken](https://github.com/openai/tiktoken) for inference) and [`scripts/tok_train.py`](https://github.com/leideng/nanochat-ascend/blob/main/scripts/tok_train.py) (stream FineWeb-Edu, run BPE, save `tokenizer.pkl` and `token_bytes.pt`).

The main hyperparameter I changed versus Karpathy’s public **$2^{16}$** runs is vocabulary size: I trained at **$2^{15} = 32\,768$** tokens for d20/d32 to shrink the embedding table and `lm_head` (see §1.1). The trained weights are on [Hugging Face: nanochat-ascend-tokenizer](https://huggingface.co/leideng/nanochat-ascend-tokenizer).

#### Training

I run tokenizer training before any GPT pretraining. From the repo root:

```bash
bash runs/run_tok_train.sh
```

[`run_tok_train.sh`](https://github.com/leideng/nanochat-ascend/blob/ee840874991bafe039f0984aaf1e34bb348f4a7e/runs/run_base_train.sh) calls [`tok_train.py`](https://github.com/leideng/nanochat-ascend/blob/ee840874991bafe039f0984aaf1e34bb348f4a7e/scripts/base_train.py) with the following options

| Flag | Default in `tok_train.py` | Notes |
|------|---------------------------|--------|
| `--max-chars` | `20_000_000_000` | Cap on characters seen during BPE training |
| `--doc-cap` | `10_000` | Max characters per document |
| `--vocab-size` | `32768` | I set this to **32K** (Karpathy’s public runs use **64K**) |

On my A3 machine with 4*80=320 logic CPUs, this took **765 s (~12.8 min)**; I logged the run in [`dev/tok_eval_results/tokenizer-training.md`](https://github.com/leideng/nanochat-ascend/blob/ee840874991bafe039f0984aaf1e34bb348f4a7e/dev/tok_eval_results/tokenizer-training.md):

| Metric | My run |
|--------|--------|
| Training time | 765.2 s |
| `vocab_size` | 32,768 |
| Special tokens | 9 |
| Token byte length (non-special) | min 1, max 19, mean 6.60, std 2.82 |

Also, here is the full training log for reference
<details markdown="1">
<summary>Tokenizer Training Log</summary>

```text
(nanochat-ascend) root@liteserver-910c-1-00001:/data/ldeng/code/nanochat-ascend# bash runs/run_tok_train.sh
OMP_NUM_THREADS is set to be: 1
NANOCHAT_CONFIG is set to be: configs/global.yaml
All runtime config is loaded from configs/global.yaml
Please run "echo $OMP_NUM_THREADS" in your terminal to see the value of OMP_NUM_THREADS environment variable
Please run "echo $NANOCHAT_CONFIG" in your terminal to see the value of NANOCHAT_CONFIG environment variable
Training the tokenizer...
max_chars: 20,000,000,000
doc_cap: 10,000
vocab_size: 32,768
2026-03-29 16:29:07,679 - rustbpe - INFO - Processing sequences from iterator (buffer_size: 8192)
2026-03-29 16:41:10,804 - rustbpe - INFO - Processed 5319189 sequences total, 9464822 unique
2026-03-29 16:41:11,186 - rustbpe - INFO - Starting BPE training: 32503 merges to compute
2026-03-29 16:41:11,186 - rustbpe - INFO - Computing initial pair counts from 9464822 unique sequences
2026-03-29 16:41:20,066 - rustbpe - INFO - Building heap with 21816 unique pairs
2026-03-29 16:41:20,068 - rustbpe - INFO - Starting merge loop
2026-03-29 16:41:34,314 - rustbpe - INFO - Progress: 1% (326/32503 merges) - Last merge: (284, 404) -> 581 (frequency: 5866412)
2026-03-29 16:41:36,513 - rustbpe - INFO - Progress: 2% (651/32503 merges) - Last merge: (304, 285) -> 906 (frequency: 2560503)
2026-03-29 16:41:37,435 - rustbpe - INFO - Progress: 3% (976/32503 merges) - Last merge: (45, 115) -> 1231 (frequency: 1545336)
2026-03-29 16:41:38,264 - rustbpe - INFO - Progress: 4% (1301/32503 merges) - Last merge: (347, 109) -> 1556 (frequency: 1103263)
2026-03-29 16:41:38,811 - rustbpe - INFO - Progress: 5% (1626/32503 merges) - Last merge: (1721, 614) -> 1881 (frequency: 851938)
2026-03-29 16:41:39,370 - rustbpe - INFO - Progress: 6% (1951/32503 merges) - Last merge: (290, 437) -> 2206 (frequency: 674393)
2026-03-29 16:41:39,900 - rustbpe - INFO - Progress: 7% (2276/32503 merges) - Last merge: (972, 1143) -> 2531 (frequency: 559880)
2026-03-29 16:41:40,263 - rustbpe - INFO - Progress: 8% (2601/32503 merges) - Last merge: (1434, 269) -> 2856 (frequency: 467260)
2026-03-29 16:41:40,883 - rustbpe - INFO - Progress: 9% (2926/32503 merges) - Last merge: (271, 668) -> 3181 (frequency: 398639)
2026-03-29 16:41:41,185 - rustbpe - INFO - Progress: 10% (3251/32503 merges) - Last merge: (1386, 271) -> 3506 (frequency: 350325)
2026-03-29 16:41:41,525 - rustbpe - INFO - Progress: 11% (3576/32503 merges) - Last merge: (395, 101) -> 3831 (frequency: 306588)
2026-03-29 16:41:41,835 - rustbpe - INFO - Progress: 12% (3901/32503 merges) - Last merge: (65, 2393) -> 4156 (frequency: 273196)
2026-03-29 16:41:42,013 - rustbpe - INFO - Progress: 13% (4226/32503 merges) - Last merge: (2180, 663) -> 4481 (frequency: 245849)
2026-03-29 16:41:42,253 - rustbpe - INFO - Progress: 14% (4551/32503 merges) - Last merge: (801, 1814) -> 4806 (frequency: 221889)
2026-03-29 16:41:42,634 - rustbpe - INFO - Progress: 15% (4876/32503 merges) - Last merge: (327, 116) -> 5131 (frequency: 201480)
2026-03-29 16:41:42,908 - rustbpe - INFO - Progress: 16% (5201/32503 merges) - Last merge: (305, 487) -> 5456 (frequency: 184870)
2026-03-29 16:41:43,334 - rustbpe - INFO - Progress: 17% (5526/32503 merges) - Last merge: (324, 258) -> 5781 (frequency: 169965)
2026-03-29 16:41:43,537 - rustbpe - INFO - Progress: 18% (5851/32503 merges) - Last merge: (3371, 3880) -> 6106 (frequency: 156282)
2026-03-29 16:41:43,823 - rustbpe - INFO - Progress: 19% (6176/32503 merges) - Last merge: (1170, 324) -> 6431 (frequency: 143670)
2026-03-29 16:41:44,076 - rustbpe - INFO - Progress: 20% (6501/32503 merges) - Last merge: (490, 282) -> 6756 (frequency: 134096)
2026-03-29 16:41:44,298 - rustbpe - INFO - Progress: 21% (6826/32503 merges) - Last merge: (365, 2224) -> 7081 (frequency: 124935)
2026-03-29 16:41:44,462 - rustbpe - INFO - Progress: 22% (7151/32503 merges) - Last merge: (2156, 324) -> 7406 (frequency: 116601)
2026-03-29 16:41:44,627 - rustbpe - INFO - Progress: 23% (7476/32503 merges) - Last merge: (286, 337) -> 7731 (frequency: 108853)
2026-03-29 16:41:44,835 - rustbpe - INFO - Progress: 24% (7801/32503 merges) - Last merge: (3947, 774) -> 8056 (frequency: 102349)
2026-03-29 16:41:44,957 - rustbpe - INFO - Progress: 25% (8126/32503 merges) - Last merge: (1164, 1813) -> 8381 (frequency: 95982)
2026-03-29 16:41:45,133 - rustbpe - INFO - Progress: 26% (8451/32503 merges) - Last merge: (83, 1408) -> 8706 (frequency: 90149)
2026-03-29 16:41:45,285 - rustbpe - INFO - Progress: 27% (8776/32503 merges) - Last merge: (313, 354) -> 9031 (frequency: 84829)
2026-03-29 16:41:45,514 - rustbpe - INFO - Progress: 28% (9101/32503 merges) - Last merge: (1727, 1888) -> 9356 (frequency: 80536)
2026-03-29 16:41:45,725 - rustbpe - INFO - Progress: 29% (9426/32503 merges) - Last merge: (2366, 705) -> 9681 (frequency: 76279)
2026-03-29 16:41:45,850 - rustbpe - INFO - Progress: 30% (9751/32503 merges) - Last merge: (82, 544) -> 10006 (frequency: 72772)
2026-03-29 16:41:46,011 - rustbpe - INFO - Progress: 31% (10076/32503 merges) - Last merge: (863, 1871) -> 10331 (frequency: 68813)
2026-03-29 16:41:46,187 - rustbpe - INFO - Progress: 32% (10401/32503 merges) - Last merge: (478, 326) -> 10656 (frequency: 65415)
2026-03-29 16:41:46,301 - rustbpe - INFO - Progress: 33% (10726/32503 merges) - Last merge: (303, 433) -> 10981 (frequency: 62237)
2026-03-29 16:41:46,450 - rustbpe - INFO - Progress: 34% (11052/32503 merges) - Last merge: (326, 99) -> 11307 (frequency: 59387)
2026-03-29 16:41:46,546 - rustbpe - INFO - Progress: 35% (11377/32503 merges) - Last merge: (2946, 176) -> 11632 (frequency: 56799)
2026-03-29 16:41:46,642 - rustbpe - INFO - Progress: 36% (11702/32503 merges) - Last merge: (98, 293) -> 11957 (frequency: 54105)
2026-03-29 16:41:46,776 - rustbpe - INFO - Progress: 37% (12027/32503 merges) - Last merge: (296, 2412) -> 12282 (frequency: 51849)
2026-03-29 16:41:46,882 - rustbpe - INFO - Progress: 38% (12352/32503 merges) - Last merge: (1577, 3333) -> 12607 (frequency: 49945)
2026-03-29 16:41:47,016 - rustbpe - INFO - Progress: 39% (12677/32503 merges) - Last merge: (275, 9539) -> 12932 (frequency: 48110)
2026-03-29 16:41:47,143 - rustbpe - INFO - Progress: 40% (13002/32503 merges) - Last merge: (429, 530) -> 13257 (frequency: 46306)
2026-03-29 16:41:47,284 - rustbpe - INFO - Progress: 41% (13327/32503 merges) - Last merge: (422, 86) -> 13582 (frequency: 44570)
2026-03-29 16:41:47,409 - rustbpe - INFO - Progress: 42% (13652/32503 merges) - Last merge: (347, 3739) -> 13907 (frequency: 42925)
2026-03-29 16:41:47,535 - rustbpe - INFO - Progress: 43% (13977/32503 merges) - Last merge: (6830, 4738) -> 14232 (frequency: 41406)
2026-03-29 16:41:47,677 - rustbpe - INFO - Progress: 44% (14302/32503 merges) - Last merge: (431, 122) -> 14557 (frequency: 39868)
2026-03-29 16:41:47,802 - rustbpe - INFO - Progress: 45% (14627/32503 merges) - Last merge: (3427, 280) -> 14882 (frequency: 38495)
2026-03-29 16:41:47,899 - rustbpe - INFO - Progress: 46% (14952/32503 merges) - Last merge: (104, 110) -> 15207 (frequency: 37165)
2026-03-29 16:41:48,038 - rustbpe - INFO - Progress: 47% (15277/32503 merges) - Last merge: (1597, 276) -> 15532 (frequency: 35934)
2026-03-29 16:41:48,130 - rustbpe - INFO - Progress: 48% (15602/32503 merges) - Last merge: (1829, 100) -> 15857 (frequency: 34713)
2026-03-29 16:41:48,278 - rustbpe - INFO - Progress: 49% (15927/32503 merges) - Last merge: (5180, 77) -> 16182 (frequency: 33531)
2026-03-29 16:41:48,559 - rustbpe - INFO - Progress: 50% (16252/32503 merges) - Last merge: (5538, 10602) -> 16507 (frequency: 32485)
2026-03-29 16:41:48,712 - rustbpe - INFO - Progress: 51% (16577/32503 merges) - Last merge: (752, 507) -> 16832 (frequency: 31442)
2026-03-29 16:41:48,799 - rustbpe - INFO - Progress: 52% (16902/32503 merges) - Last merge: (2727, 522) -> 17157 (frequency: 30430)
2026-03-29 16:41:48,908 - rustbpe - INFO - Progress: 53% (17227/32503 merges) - Last merge: (365, 66) -> 17482 (frequency: 29533)
2026-03-29 16:41:48,987 - rustbpe - INFO - Progress: 54% (17552/32503 merges) - Last merge: (9505, 1129) -> 17807 (frequency: 28651)
2026-03-29 16:41:49,169 - rustbpe - INFO - Progress: 55% (17877/32503 merges) - Last merge: (2532, 1069) -> 18132 (frequency: 27817)
2026-03-29 16:41:49,262 - rustbpe - INFO - Progress: 56% (18202/32503 merges) - Last merge: (357, 2716) -> 18457 (frequency: 27052)
2026-03-29 16:41:49,335 - rustbpe - INFO - Progress: 57% (18527/32503 merges) - Last merge: (16694, 537) -> 18782 (frequency: 26322)
2026-03-29 16:41:49,449 - rustbpe - INFO - Progress: 58% (18852/32503 merges) - Last merge: (410, 470) -> 19107 (frequency: 25552)
2026-03-29 16:41:49,532 - rustbpe - INFO - Progress: 59% (19177/32503 merges) - Last merge: (19297, 3752) -> 19432 (frequency: 24842)
2026-03-29 16:41:49,623 - rustbpe - INFO - Progress: 60% (19502/32503 merges) - Last merge: (6358, 466) -> 19757 (frequency: 24121)
2026-03-29 16:41:49,699 - rustbpe - INFO - Progress: 61% (19827/32503 merges) - Last merge: (3532, 3793) -> 20082 (frequency: 23511)
2026-03-29 16:41:49,767 - rustbpe - INFO - Progress: 62% (20152/32503 merges) - Last merge: (5579, 316) -> 20407 (frequency: 22890)
2026-03-29 16:41:49,845 - rustbpe - INFO - Progress: 63% (20477/32503 merges) - Last merge: (282, 698) -> 20732 (frequency: 22272)
2026-03-29 16:41:49,917 - rustbpe - INFO - Progress: 64% (20802/32503 merges) - Last merge: (439, 16542) -> 21057 (frequency: 21649)
2026-03-29 16:41:49,979 - rustbpe - INFO - Progress: 65% (21127/32503 merges) - Last merge: (109, 1032) -> 21382 (frequency: 21099)
2026-03-29 16:41:50,088 - rustbpe - INFO - Progress: 66% (21452/32503 merges) - Last merge: (5115, 9456) -> 21707 (frequency: 20591)
2026-03-29 16:41:50,163 - rustbpe - INFO - Progress: 67% (21778/32503 merges) - Last merge: (16168, 426) -> 22033 (frequency: 20115)
2026-03-29 16:41:50,233 - rustbpe - INFO - Progress: 68% (22103/32503 merges) - Last merge: (67, 991) -> 22358 (frequency: 19566)
2026-03-29 16:41:50,342 - rustbpe - INFO - Progress: 69% (22428/32503 merges) - Last merge: (88, 88) -> 22683 (frequency: 19069)
2026-03-29 16:41:50,400 - rustbpe - INFO - Progress: 70% (22753/32503 merges) - Last merge: (336, 20903) -> 23008 (frequency: 18583)
2026-03-29 16:41:50,456 - rustbpe - INFO - Progress: 71% (23078/32503 merges) - Last merge: (8303, 1217) -> 23333 (frequency: 18135)
2026-03-29 16:41:50,513 - rustbpe - INFO - Progress: 72% (23403/32503 merges) - Last merge: (2787, 282) -> 23658 (frequency: 17686)
2026-03-29 16:41:50,603 - rustbpe - INFO - Progress: 73% (23728/32503 merges) - Last merge: (3910, 376) -> 23983 (frequency: 17285)
2026-03-29 16:41:50,682 - rustbpe - INFO - Progress: 74% (24053/32503 merges) - Last merge: (20993, 469) -> 24308 (frequency: 16851)
2026-03-29 16:41:50,785 - rustbpe - INFO - Progress: 75% (24378/32503 merges) - Last merge: (9470, 569) -> 24633 (frequency: 16445)
2026-03-29 16:41:50,861 - rustbpe - INFO - Progress: 76% (24703/32503 merges) - Last merge: (2272, 263) -> 24958 (frequency: 16059)
2026-03-29 16:41:50,918 - rustbpe - INFO - Progress: 77% (25028/32503 merges) - Last merge: (562, 16963) -> 25283 (frequency: 15699)
2026-03-29 16:41:50,984 - rustbpe - INFO - Progress: 78% (25353/32503 merges) - Last merge: (388, 13424) -> 25608 (frequency: 15356)
2026-03-29 16:41:51,062 - rustbpe - INFO - Progress: 79% (25678/32503 merges) - Last merge: (586, 587) -> 25933 (frequency: 15022)
2026-03-29 16:41:51,123 - rustbpe - INFO - Progress: 80% (26003/32503 merges) - Last merge: (16906, 9119) -> 26258 (frequency: 14701)
2026-03-29 16:41:51,225 - rustbpe - INFO - Progress: 81% (26328/32503 merges) - Last merge: (84, 475) -> 26583 (frequency: 14382)
2026-03-29 16:41:51,289 - rustbpe - INFO - Progress: 82% (26653/32503 merges) - Last merge: (7152, 4562) -> 26908 (frequency: 14067)
2026-03-29 16:41:51,352 - rustbpe - INFO - Progress: 83% (26978/32503 merges) - Last merge: (10020, 419) -> 27233 (frequency: 13787)
2026-03-29 16:41:51,415 - rustbpe - INFO - Progress: 84% (27303/32503 merges) - Last merge: (2335, 282) -> 27558 (frequency: 13515)
2026-03-29 16:41:51,539 - rustbpe - INFO - Progress: 85% (27628/32503 merges) - Last merge: (10737, 316) -> 27883 (frequency: 13220)
2026-03-29 16:41:51,593 - rustbpe - INFO - Progress: 86% (27953/32503 merges) - Last merge: (446, 6760) -> 28208 (frequency: 12959)
2026-03-29 16:41:51,654 - rustbpe - INFO - Progress: 87% (28278/32503 merges) - Last merge: (377, 342) -> 28533 (frequency: 12691)
2026-03-29 16:41:51,750 - rustbpe - INFO - Progress: 88% (28603/32503 merges) - Last merge: (15546, 21648) -> 28858 (frequency: 12435)
2026-03-29 16:41:51,812 - rustbpe - INFO - Progress: 89% (28928/32503 merges) - Last merge: (8279, 600) -> 29183 (frequency: 12182)
2026-03-29 16:41:51,901 - rustbpe - INFO - Progress: 90% (29253/32503 merges) - Last merge: (446, 2688) -> 29508 (frequency: 11936)
2026-03-29 16:41:51,947 - rustbpe - INFO - Progress: 91% (29578/32503 merges) - Last merge: (4938, 825) -> 29833 (frequency: 11710)
2026-03-29 16:41:52,003 - rustbpe - INFO - Progress: 92% (29903/32503 merges) - Last merge: (407, 2106) -> 30158 (frequency: 11510)
2026-03-29 16:41:52,067 - rustbpe - INFO - Progress: 93% (30228/32503 merges) - Last merge: (37, 1570) -> 30483 (frequency: 11288)
2026-03-29 16:41:52,117 - rustbpe - INFO - Progress: 94% (30553/32503 merges) - Last merge: (9668, 121) -> 30808 (frequency: 11083)
2026-03-29 16:41:52,161 - rustbpe - INFO - Progress: 95% (30878/32503 merges) - Last merge: (5304, 1808) -> 31133 (frequency: 10871)
2026-03-29 16:41:52,199 - rustbpe - INFO - Progress: 96% (31203/32503 merges) - Last merge: (4559, 280) -> 31458 (frequency: 10679)
2026-03-29 16:41:52,282 - rustbpe - INFO - Progress: 97% (31528/32503 merges) - Last merge: (20625, 93) -> 31783 (frequency: 10492)
2026-03-29 16:41:52,347 - rustbpe - INFO - Progress: 98% (31853/32503 merges) - Last merge: (195, 176) -> 32108 (frequency: 10319)
2026-03-29 16:41:52,383 - rustbpe - INFO - Progress: 99% (32178/32503 merges) - Last merge: (428, 2048) -> 32433 (frequency: 10149)
2026-03-29 16:41:52,438 - rustbpe - INFO - Progress: 100% (32503/32503 merges) - Last merge: (14017, 539) -> 32758 (frequency: 9971)
2026-03-29 16:41:52,438 - rustbpe - INFO - Finished training: 32503 merges completed
Training time: 765.21s
Saved tokenizer encoding to .cache/output/tokenizer/tokenizer.pkl
Saved token_bytes to .cache/output/tokenizer/token_bytes.pt
Evaluating the tokenizer...
```
</details>


> **Note**
>
> In addition, When I debug on CPU only, I shorten `--max-chars` (e.g. `5_000_000`) so the loop finishes in minutes.

#### Evaluation

After training, we can run `tok_eval.py` to see how many bytes each token represents. We define **compression ratio** = UTF-8 bytes / token count (higher is better). Karpathy has already encoded fixed snippets (English news, Korean, Python, LaTeX, science) plus one parquet batch from the train and val splits. I compare the trained tokenizer against OpenAI baselines loaded through tiktoken: **gpt2**, **cl100k_base** (GPT-4), and **o200k_base** (GPT-5). Every case must satisfy `decode(encode(text)) == text`. **Relative diff %** is $(\text{baseline tokens} - \text{my tokens}) / \text{baseline tokens}$; positive means I use fewer tokens than the baseline.

I archived the full tables under [`dev/tok_eval_results/tokenizer-evaluation.md`](https://github.com/leideng/nanochat-ascend/blob/ee840874991bafe039f0984aaf1e34bb348f4a7e/dev/tok_eval_results/tokenizer-evaluation.md). Below is an excerpt from the log.

<details markdown="1">
<summary>Tokenizer Evaluation Log</summary>

```text
Vocab sizes:
GPT-2: 50257
GPT-4: 100277
GPT-5: 200019
Ours: 32768

Comparison with GPT-2:
===============================================================================================
Text Type  Bytes    GPT-2           Ours            Relative     Better
                    Tokens  Ratio   Tokens  Ratio   Diff %
-----------------------------------------------------------------------------------------------
news       1819     404     4.50    403     4.51       +0.2%     Ours
korean     893      745     1.20    797     1.12       -7.0%     GPT-2
code       1259     576     2.19    622     2.02       -8.0%     GPT-2
math       1834     936     1.96    1009    1.82       -7.8%     GPT-2
science    1112     260     4.28    258     4.31       +0.8%     Ours
fwe-train  4208518  900364  4.67    892491  4.72       +0.9%     Ours
fwe-val    4776536  1031472 4.63    1026892 4.65       +0.4%     Ours

Comparison with GPT-4:
===============================================================================================
Text Type  Bytes    GPT-4           Ours            Relative     Better
                    Tokens  Ratio   Tokens  Ratio   Diff %
-----------------------------------------------------------------------------------------------
news       1819     387     4.70    403     4.51       -4.1%     GPT-4
korean     893      364     2.45    797     1.12     -119.0%     GPT-4
code       1259     309     4.07    622     2.02     -101.3%     GPT-4
math       1834     832     2.20    1009    1.82      -21.3%     GPT-4
science    1112     249     4.47    258     4.31       -3.6%     GPT-4
fwe-train  4208518  874799  4.81    892491  4.72       -2.0%     GPT-4
fwe-val    4776536  1004142 4.76    1026892 4.65       -2.3%     GPT-4

Comparison with GPT-5:
===============================================================================================
Text Type  Bytes    GPT-5           Ours            Relative     Better
                    Tokens  Ratio   Tokens  Ratio   Diff %
-----------------------------------------------------------------------------------------------
news       1819     379     4.80    403     4.51       -6.3%     GPT-5
korean     893      219     4.08    797     1.12     -263.9%     GPT-5
code       1259     307     4.10    622     2.02     -102.6%     GPT-5
math       1834     836     2.19    1009    1.82      -20.7%     GPT-5
science    1112     239     4.65    258     4.31       -7.9%     GPT-5
fwe-train  4208518  865531  4.86    892491  4.72       -3.1%     GPT-5
fwe-val    4776536  992971  4.81    1026892 4.65       -3.4%     GPT-5
```
</details>

For the **English pretraining distribution I care about** (FineWeb-Edu), our trained 32K BPE tokenizer is close to OpenAI’s public encoders on bytes-per-token. However, it has worse compression on Korean, code, and LaTeX as we did not see enough such data for a 20B characters and for a limited 32K vocab size as compared to OpenAI's 50K, 100K, and 200K. But that was enough for me to proceed to GPT pretraining.

## 2. Next-token-prediction (NTP) Paradigm

### 2.1. Language Modeling

As [[6]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) points out, natural language tasks
can be formulated as an *unsupervised* multitask learning problem via language sequence prediction.
For example, a translation task can be written as the language sequence/string `(translate the given english text to french, english text, french text)`;
a reading comprehension task can be written as the language string `(answer the question based on the given document, document, question, answer)`;
a math proof problem can be written as the language string `(prove the following math problem, problem, proof)`;
and a coding problem can be written as the language string `(write a python code script to solve the following problem, problem, code)`.
Therefore, language modeling centers on language sequences/strings.


After tokenization, a language string $\vec{s} \in \Sigma^{\ast}$ becomes a token sequence of length $l$

$$
\vec{x} := (x_1, x_2, \cdots, x_l) = T_{enc}(\vec{s}) \in \mathcal{V}^{l}, \quad x_t \in \mathcal{V},
$$

with $\lvert\mathcal{V}\rvert = V$. On the other hand, any token sequence can be decoded to a language string. Thus, next we will only consider token sequences.


We can formally model token sequence set and token sequence prediction problem as follows.

A token sequence is a random vector (of random length)

$$
\vec{X} := (X_1, X_2, \dots, X_L)
$$

where each token {::nomarkdown}$X_t \in \mathcal{V}${:/nomarkdown} and the length $L$ is itself a **stopping time** (at each step $t$, whether to stop is determined by the information up to $t$).

To better define the stopping time $L$, we add a special token `<bos>` into the vocabulary, which marks the beginning of a sequence.
Note that we do not add `<eos>` special token since `<bos>` also marks the end of a sequence. Thus, `<bos>` serves both as the beginning and the end of a sequence.

Thus, a valid token sequence becomes

$$
\vec{X} = (X_1=\text{<bos>}, X_2, \dots, X_{L-1}, X_L=\text{<bos>}), \quad X_2,\dots,X_{L-1} \neq \text{<bos>}, \quad L \ge 2.
$$

In addition, due to the limited computation capability, we cannot handle arbitrary long sequences. Thus, we will impose a context window {::nomarkdown}$L_{\max}${:/nomarkdown} restriction and we require that {::nomarkdown}$L \le L_{\max}${:/nomarkdown}.
In our nanochat-ascend project, we set {::nomarkdown}$L_{\max}=2048${:/nomarkdown}. The latest model such as DeepSeek V4 has 1M context windows size, i.e., $L=1048576$.

Then the stop time $L$ is defined as either {::nomarkdown}$L = L_{\max}${:/nomarkdown} or {::nomarkdown}$X_L = \text{&lt;bos&gt;}${:/nomarkdown}, which can be determined by the information up to $t$ at each step $t$.


The set of all valid token sequences is denoted by {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown}, which serves as **the sample space**.  We can count the total number of sequences by length:

| $L$ | Free positions | # sequences |
|---|---|---|
| $2$ | $0$ | $V^0 = 1$ |
| $3$ | $1$ | $V^1$ |
| $4$ | $2$ | $V^2$ |
| $\vdots$ | $\vdots$ | $\vdots$ |
| {::nomarkdown}$L_{\max}-1${:/nomarkdown} | {::nomarkdown}$L_{\max}-3${:/nomarkdown} | {::nomarkdown}$V^{L_{\max}-3}${:/nomarkdown} |
| {::nomarkdown}$L_{\max}${:/nomarkdown} where {::nomarkdown}$X_{L_{\max}} = \text{&lt;bos&gt;}${:/nomarkdown} | {::nomarkdown}$L_{\max}-2${:/nomarkdown} | {::nomarkdown}$V^{L_{\max}-2}${:/nomarkdown} |
| {::nomarkdown}$L_{\max}${:/nomarkdown} where {::nomarkdown}$X_{L_{\max}} \neq \text{&lt;bos&gt;}${:/nomarkdown}| {::nomarkdown}$L_{\max}-1${:/nomarkdown} | {::nomarkdown}$V^{L_{\max}-1}${:/nomarkdown} |

Thus the total number of all valid sequences over all possible lengths is

$$
|\Omega_{\text{truth}}| = \sum_{i=0}^{L_{\max}-1} V^i = \frac{V^{L_{\max}}-1}{V-1} \approx V^{L_{\max}-1}
$$

We now assume that random vector {::nomarkdown}$\vec{X} \in \Omega_{\text{truth}}${:/nomarkdown} follows a **ground truth probability measure** {::nomarkdown}$P_{\text{truth}}${:/nomarkdown} on the sample space {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown}.
Since a probability measure defines the probability over any events of the sample space (i.e., any subset of the sample space),
we can then get whatever joint probabilities and marginal probabilities under measure {::nomarkdown}$P_{\text{truth}}${:/nomarkdown}.
For the current subsection, we will define {::nomarkdown}$P_{\text{truth}}${:/nomarkdown} as a usual probability distribution over each element in the sample space.
First, the probability of 

$$
\vec{X} = \vec{\omega} = (\omega_1,\omega_2,\cdots, \omega_l) \in \Omega_{\text{truth}}
$$

is {::nomarkdown}$P_{\text{truth}}(\vec{\omega})${:/nomarkdown}, which is the joint probability that
{::nomarkdown}$L=l, X_1=\omega_1, X_2=\omega_2, \cdots, X_l = \omega_l${:/nomarkdown}, i.e.,

$$
P_{\text{truth}}(\vec{\omega}) = P_{\text{truth}}(\vec{X} = \vec{\omega}) = P_{\text{truth}} (L=l, X_1=\omega_1, X_2=\omega_2, \cdots, X_l=\omega_l).
$$

Once we have the joint distribution {::nomarkdown}$P_{\text{truth}}(\vec{\omega})${:/nomarkdown}, i.e., we have the probability for each element in the sample space,
we can get any marginal distributions, e.g.,

$$
\begin{aligned}
P_{\text{truth}}(X_1=a)
&:= \sum_{\substack{\vec{\omega} = (\omega_1,\omega_2,\cdots,\omega_l) \in \Omega_{\text{truth}}: \omega_1=a}}
P_{\text{truth}}(\vec{\omega})
=
\begin{cases}
    1, & \text{if } a=\text{<bos>},  \\
    0, & \text{otherwise.}
\end{cases}
\end{aligned}
$$

$$
\begin{aligned}
&P_{\text{truth}}(X_1=a,X_2=b,X_3=c) 
:= \sum_{\substack{\vec{\omega} = (\omega_1,\omega_2, \omega_3, \cdots,\omega_l) \in \Omega_{\text{truth}}: l \ge 3, \omega_1=a, \omega_2=b, \omega_3=c}}
P_{\text{truth}}(\vec{\omega}).
\end{aligned}
$$


$$
\begin{aligned}
&P_{\text{truth}}(X_3=c) 
:= \sum_{\substack{\vec{\omega} = (\omega_1,\omega_2, \omega_3, \cdots,\omega_l) \in \Omega_{\text{truth}}: l \ge 3, \omega_3=c}}
P_{\text{truth}}(\vec{\omega})
\end{aligned}
$$

For notation simplicity, we sometimes omit the random variable {::nomarkdown}$X_i${:/nomarkdown} but simply use {::nomarkdown}$\omega_i/x_i/y_i${:/nomarkdown} when the context is clear to show that {::nomarkdown}$\omega_i/x_i/y_i${:/nomarkdown} is the $i$-th token in a sequence, namely

$$
P_{\text{truth}}(\omega_i) := P_{\text{truth}} (X_i=\omega_i).
$$

$$
P_{\text{truth}}(x_i) := P_{\text{truth}} (X_i=x_i).
$$

$$
P_{\text{truth}}(y_i) := P_{\text{truth}} (X_i=y_i).
$$

Similarly, we also sometimes omit several random variables to denote the probability for several tokens.
For example, if we define {::nomarkdown}$\vec{x}=(x_1,x_2,x_3)${:/nomarkdown} and {::nomarkdown}$\vec{y}=(y_4,y_5,y_6)${:/nomarkdown}, we have

$$
P_{\text{truth}}(\vec{x}) := P_{\text{truth}} (X_1=x_1, X_2=x_2, X_3=x_3).
$$

$$
P_{\text{truth}}(\vec{y}) := P_{\text{truth}} (X_4=y_4, X_5=y_5, X_6=y_6).
$$

$$
P_{\text{truth}}(\vec{y} \mid \vec{x}) := P_{\text{truth}} (X_4=y_4, X_5=y_5, X_6=y_6 \mid X_1=x_1, X_2=x_2, X_3=x_3).
$$

$$
P_{\text{truth}}(\vec{x},y_4) := P_{\text{truth}} (X_1=x_1,X_2=x_2,X_3=x_3,X_4=y_4).
$$

$$
P_{\text{truth}}(y_5, \vec{x}) := P_{\text{truth}}(\vec{x}, y_5) = P_{\text{truth}} (X_1=x_1,X_2=x_2,X_3=x_3,X_5=y_5).
$$

$$
P_{\text{truth}}(\vec{x},\vec{y}) := P_{\text{truth}} (X_1=x_1,X_2=x_2,X_3=x_3,X_4=y_4,X_5=y_5,X_6=y_6).
$$


Under this theoretical framework, we can define language tasks now: a language task is to predict the remaining part (aka response) based on the given preceding part (aka prompt) of a token sequence. This is exactly a Seq2Seq problem [[7]](https://arxiv.org/pdf/1409.3215). Namely, given the prompt (input) {::nomarkdown}$\vec{x}=(x_1,x_2,\cdots,x_{i-1})${:/nomarkdown} where $i \ge 1$, we need to predict the response (output) {::nomarkdown}$\vec{y}=(y_i,y_{i+1},\cdots, y_{l})${:/nomarkdown}
such that the concatenation sequence {::nomarkdown}$(\vec{x},\vec{y})=(x_1,x_2,\cdots,x_{i-1},y_i,y_{i+1},\cdots, y_{l})${:/nomarkdown} is
the valid token sequence in {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown} with the highest probability (ties break arbitrarily), i.e.,

$$
\begin{aligned}
\vec{y^*}
&:= \arg\max_{\vec{y}: (\vec{x},\vec{y}) \in \Omega_{\text{truth}}}  P_{\text{truth}}( \vec{x},\vec{y} ) 
= \arg\max_{\vec{y}: (\vec{x},\vec{y}) \in \Omega_{\text{truth}}}  P_{\text{truth}} ( \vec{y} \mid \vec{x} ).
\end{aligned}
$$

Note that this is the exact maximum a posteriori (MAP) decoding problem to predict the most likely sequence [[8]](https://aclanthology.org/2022.tacl-1.58/). In practice, we often use greedy decoding/beam search/top-k or top-p sampling to approximately solve this problem.

A naive approach to solve this language task problem is to directly estimate {::nomarkdown}$P_{\text{truth}}(\vec{\omega})${:/nomarkdown} for any {::nomarkdown}$\vec{\omega} \in \Omega_{\text{truth}}${:/nomarkdown}.
However, this is intractable due to the astronomically large sample space {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown}.
If $V \sim 10^5$ and {::nomarkdown}$L_{\max} \sim 10^3${:/nomarkdown}, then the sample space is of size {::nomarkdown}$|\Omega_{\text{truth}}| \approx V^{L_{\max}-1} \sim 10^{5000}${:/nomarkdown}.

Therefore, estimating $P(\vec{\omega})$ (the joint probability) as a flat categorical distribution is hopeless.
We seek a lightweight factorized solution, which is the famous next-token prediction (NTP) paradigm.

### 2.2. Theory of NTP

By the chain rule, we have

$$
\begin{aligned}
P_{\text{truth}} ( \vec{y} \mid \vec{x} )
&= \prod_{t=i}^{l} P_{\text{truth}} (y_t \mid x_1,x_2,\cdots,x_{i-1}, y_i, \cdots, y_{t-1}),  \qquad i \ge 1
\end{aligned}
$$

with the convention {::nomarkdown}$P_{\text{truth}}(X_1 = \text{&lt;bos&gt;} \mid \emptyset) = 1${:/nomarkdown}.
Now we have factorized the conditional probability {::nomarkdown}$P_{\text{truth}} ( \vec{y} \mid \vec{x} )${:/nomarkdown} into $l-i+1$ next-token
conditional probabilities 

$$
P_{\text{truth}} (y_t \mid x_1,x_2,\cdots,x_{i-1}, y_i, \cdots, y_{t-1}).
$$


To proceed further analysis, let us define

$$
\begin{aligned}
\vec{\omega}_{t} := (\omega_1,\omega_2,\cdots,\omega_{t-1}, \omega_{t}) \\
\vec{\omega}_{\lt t} := \vec{\omega}_{t-1} = (\omega_1,\omega_2,\cdots,\omega_{t-1})
\end{aligned}
$$

with the convention {::nomarkdown}$\vec{\omega}_{\lt 1}=\vec{\omega}_{0} = \emptyset${:/nomarkdown}.

In general, {::nomarkdown}$P_{\text{truth}} (\omega_t \mid \vec{\omega}_{\lt t})${:/nomarkdown} depends on future tokens after position $t$. Namely, we should compute it as follows,

$$
\begin{aligned}
P_{\text{truth}} (\omega_t \mid \vec{\omega}_{\lt t})
= \frac{P_{\text{truth}} (\vec{\omega}_{\lt t}, \omega_t)}{P_{\text{truth}} (\vec{\omega}_{\lt t})} 
= \frac{
\sum\limits_{\substack{\vec{\nu} \in \Omega_{\text{truth}}: \vec{\nu}_{\lt t} = \vec{\omega}_{\lt t},\, \nu_t=\omega_t}}
P_{\text{truth}} (\vec{\nu})
}
{
\sum\limits_{\substack{\vec{\nu} \in \Omega_{\text{truth}}: \vec{\nu}_{\lt t} = \vec{\omega}_{\lt t}}}
P_{\text{truth}} (\vec{\nu})
}.
\end{aligned}
$$

Therefore, chain-rule factorization does not reduce the computational or modeling complexity. We now make a fundamental causal assumption,
which serves as the foundation for next-token prediction paradigm.

**Causal Assumption**. Each token's probability is conditioned only on all preceding tokens, never on future tokens. Namely, we have

$$
\begin{aligned}
&P_{\text{truth}} (\omega_t \mid \vec{\omega}_{\lt t}) = P_{\text{truth}} (\omega_t \mid \vec{\omega}_{\lt t}, \omega_{t+1}, \omega_{t+2}, \cdots, \omega_l), \\
&\quad \forall \vec{\omega}=(\vec{\omega}_{\lt t}, \omega_t, \omega_{t+1}, \omega_{t+2}, \cdots, \omega_l) \in \Omega_{\text{truth}}, 
\quad \forall t=1,2,\cdots, L_{\max}.
\end{aligned}
$$

Before we show this autoregressive factorization significantly reduces the computation complexity, we should provide justification to make this causal assumption. One key reason is that we humans **generally** read languages from the left to right in a linear manner. We gradually gain more information when we read the $t$-th token only based on the preceding tokens before position $t$, but not based on future tokens after position $t$. I have emphasized the term **"generally"** because the linear manner does not necessarily hold for sure. There are some other paradigms for language models, including Bert which predicts token based on the preceding and future tokens (aka, fill in the middle) and diffusion-based LLMs (dLLMs) which directly predict the whole sequence without considering left-to-right order.


Under this causal assumption, we can greatly simplify the probability measure {::nomarkdown}$P_{\text{truth}}${:/nomarkdown} as follows. For every {::nomarkdown}$\vec{\omega} = (\omega_1, \omega_2, \cdots, \omega_l) \in \Omega_{\text{truth}}${:/nomarkdown}, we have

$$
P_{\text{truth}} (\vec{\omega}) = \prod_{t=1}^{l} P_{\text{truth}} (\omega_t \mid \vec{w}_{\lt t}).
$$

Since {::nomarkdown}$P_{\text{truth}} (\omega_t \mid \vec{w}_{\lt t})${:/nomarkdown} does not depend on tokens after position $t$, we can model/estimate it only based on all length-$t$ sequences and all **length-$t$ prefix subsequences of sequences whose lengths are larger than $t$** in {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown}, denoted by {::nomarkdown}$\Omega_{\text{truth}, t}${:/nomarkdown}, which reduces the computation/modeling complexity. Rigorously, {::nomarkdown}$\Omega_{\text{truth}, t}${:/nomarkdown} is defined as

$$
\begin{aligned}
\Omega_{\text{truth}, t} = \Big\lbrace
\vec{\omega}_t = (\omega_1, \omega_2, \cdots, \omega_t): 
\vec{w}=(\vec{\omega}_t, \omega_{t+1}, \cdots, \omega_{l}) \in \Omega_{\text{truth}}, l \ge t
\Big\rbrace.
\end{aligned}
$$

Note that again {::nomarkdown}$\Omega_{\text{truth}, t}${:/nomarkdown} is not all length-$t$ sequences but all length-$t$ prefix subsequences in {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown}, which will affect how to perform pretraining.


We now define the length-$t$
probability measure {::nomarkdown}$P_{\text{truth},t}${:/nomarkdown} on the sample space {::nomarkdown}$\Omega_{\text{truth}, t}${:/nomarkdown}, i.e.,

$$
\begin{aligned}
P_{\text{truth},t}(\vec{\omega}_t = (\omega_1, \omega_2, \cdots, \omega_t)) 
= P_{\text{truth}}(\omega_1, \omega_2, \cdots, \omega_t), \qquad  1 \le t \le L_{\max}
\end{aligned}
$$

Then, we have

$$
\begin{aligned}
P_{\text{truth}} (\vec{\omega})
= \prod_{t=1}^{l} P_{\text{truth}} (\omega_t \mid \vec{w}_{\lt t}) 
= \prod_{t=1}^{l} P_{\text{truth},t} (\omega_t \mid \vec{w}_{\lt t}).
\end{aligned}
$$

Therefore, to estimate the sequence-level global probability measures {::nomarkdown}$P_{\text{truth}}${:/nomarkdown} is reduced to estimate the token-level local probability measures {::nomarkdown}$P_{\text{truth},1}${:/nomarkdown}, {::nomarkdown}$P_{\text{truth},2}${:/nomarkdown}, $\cdots$, and {::nomarkdown}$P_{\text{truth},L_{\max}}${:/nomarkdown}, each of which becomes much easier with equal-length inputs. Once we have a good estimation for all such {::nomarkdown}$P_{\text{truth},t}${:/nomarkdown}, we can solve the language task problem by selecting $\vec{y}^*$ to maximize

$$
\begin{aligned}
P_{\text{truth}} ( \vec{y} | \vec{x} )
&= \prod_{t=i}^{l} P_{\text{truth}} (y_t \mid x_1,x_2,\cdots,x_{i-1}, y_i, \cdots, y_{t-1}) \\
&= \prod_{t=i}^{l} P_{\text{truth},t} (y_t \mid x_1,x_2,\cdots,x_{i-1}, y_i, \cdots, y_{t-1}), \qquad i \ge 1
\end{aligned}
$$

To facilitate the further analysis, we also define the sample space of all prefix subsequences of data samples of length at least $t$, i.e.,


$$
\begin{aligned}
\Omega_{\text{truth}, \lt t} = \Big\lbrace
\vec{\omega}_{\lt t} = (\omega_1, \omega_2, \cdots, \omega_{t-1}): 
\vec{w}=(\vec{\omega}_{t-1}, \omega_t, \omega_{t+1}, \cdots, \omega_{l}) \in \Omega_{\text{truth}}, l \ge t
\Big\rbrace.
\end{aligned}
$$

Note that {::nomarkdown}$\Omega_{\text{truth}, \lt t} \neq \Omega_{\text{truth}, t-1}${:/nomarkdown} since  {::nomarkdown}$\Omega_{\text{truth}, \lt t}${:/nomarkdown} still considers all data samples of length at least $t$ while {::nomarkdown}$\Omega_{\text{truth}, t-1}${:/nomarkdown} considers all data samples of at least $t-1$.



Directly estimating {::nomarkdown}$P_{\text{truth},t}(\vec{\omega}_t)${:/nomarkdown} is still of exponential complexity since {::nomarkdown}$| \Omega_{\text{truth}, t} | = (V+1)V^{t-2}${:/nomarkdown} for $t \ge 2$.
From the above equation, we do not need to know {::nomarkdown}$P_{\text{truth},t}(\vec{\omega}_t)${:/nomarkdown}. Instead, we only need to know {::nomarkdown}$P_{\text{truth},t} (\omega_t \mid \vec{w}_{\lt t})${:/nomarkdown}. Namely, we only need to **predict the next token based on all preceding tokens iteratively**, which is exactly the NTP paradigm.

We only need $V+1$ numbers to represent {::nomarkdown}$P_{\text{truth},t} (\omega_t \mid \vec{w}_{\lt t})${:/nomarkdown} for any given preceding {::nomarkdown}$\vec{w}_{\lt t}${:/nomarkdown}, i.e.,

$$
\begin{aligned}
P_{\text{truth},t} (\omega_t = 1 \mid \vec{w}_{\lt t}), \quad \cdots, \quad P_{\text{truth},t} (\omega_t = V \mid \vec{w}_{\lt t}), 
\quad P_{\text{truth},t} (\omega_t = \text{<bos>}  \mid \vec{w}_{\lt t}).
\end{aligned}
$$

We can easily design a neural network with parameters {::nomarkdown}$\theta_t${:/nomarkdown} to learn this **conditional next-token distribution** {::nomarkdown}$P_{\theta_t}(\,\cdot\mid \vec{\omega}_{\lt t})${:/nomarkdown} which
takes a token sequence of length $t-1$ as an input and outputs a probability distribution over the vocabulary of length $V+1$. In this way, we  need {::nomarkdown}$L_{\max}${:/nomarkdown} neural networks for all positions, which still brings complexity.
To reduce complexity further, we require all conditional next-token distributions to share the same parameters $\theta$. This requires an architecture to take input of arbitrary length while elegantly capturing the next token dependence. Recurrent networks achieved this via shared recurrence; modern decoder-only transformers achieve it via self-attention mechanism. We will describe the GPT neural networks in detail later.

Though the conditional probabilities on all positions share the same parameter $\theta$, we still need to find a way to optimize $\theta$ in the global sense.
Clearly, the optimal $\theta$ to {::nomarkdown}$P_{\theta_t}(\,\cdot\mid \vec{\omega}_{\lt t})${:/nomarkdown} for predicting token $t$ does not mean it is optimal to {::nomarkdown}$P_{\theta_{t+1}}(\,\cdot\mid \vec{\omega}_{\lt t+1})${:/nomarkdown}
for predicting token $t+1$.
The natural way to solve this **token-level** local optimality problem is to look at our original **sequence-level** problem. That is to say,
we need to find a neural network with parameter $\theta$ such that

$$
P_{\theta} ( \vec{y} \mid \vec{x} ) = \prod_{t=i}^{l} P_{\theta} (y_t \mid x_1,x_2,\cdots,x_{i-1}, y_i, \cdots, y_{t-1}),
$$

is as close to {::nomarkdown}$P_{\text{truth}} ( \vec{y} \mid \vec{x} )${:/nomarkdown} as possible.

We also call {::nomarkdown}$P_{\theta}${:/nomarkdown} the probability measure over the sample space {::nomarkdown}$\Omega_{\theta}${:/nomarkdown} determined by neural network with parameter $\theta$. Equivalently,
we also aim at {::nomarkdown}$P_{\theta}${:/nomarkdown} over {::nomarkdown}$\Omega_{\theta}${:/nomarkdown} as close to {::nomarkdown}$P_{\text{truth}}${:/nomarkdown} over {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown} as possible. In the next section, we will describe the theory of pretraining, which is
try to use {::nomarkdown}$P_{\theta}${:/nomarkdown} to approximate {::nomarkdown}$P_{\text{truth}}${:/nomarkdown} via the NTP-based token-level conditional probabilities {::nomarkdown}$P_{\theta}(\,\cdot\mid \vec{\omega}_{\lt t})${:/nomarkdown}.

## 3. Theory of Pretraining (Independent of Neural Network Architectures)

Based on the previous NTP paradigm, I will introduce the theory of pretraining, which is independent of adopted neural network architectures. We just assume
a general neural network with parameters $\theta$ which define the estimated/predicted probability measure {::nomarkdown}$P_{\theta}${:/nomarkdown} over the sample
space {::nomarkdown}$\Omega_{\theta}${:/nomarkdown}.

### 3.1. Modeling of Pretraining Dataset

To solve the problem in the previous section, we need to feed a training dataset. Let us model the training dataset in this section.

The training dataset consists of $N$ token sequences of length in the sample space {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown}.
The $n$-th sequence of length {::nomarkdown}$l_n \le L_{\max}${:/nomarkdown} is denoted as

$$
\vec{x}^n = (x^n_1, x^n_2, \cdots, x^n_{l_n}), \quad l_n \le L_{\max},
$$

which contributes once for the following *sequence-level* prompt-response pairs,

| $\vec{x}$ | $\vec{y}$ | Count |
|---|---|---|
| $\emptyset$ | {::nomarkdown}$(x^n_1, x^n_2, x^n_3, x^n_4, \cdots, x^n_{l_n})${:/nomarkdown} | 1 |
| {::nomarkdown}$(x^n_1)${:/nomarkdown} | {::nomarkdown}$(x^n_2, x^n_3, x^n_4, \cdots, x^n_{l_n})${:/nomarkdown} | 1 |
| {::nomarkdown}$(x^n_1, x^n_2)${:/nomarkdown} | {::nomarkdown}$(x^n_3, x^n_4, \cdots, x^n_{l_n})${:/nomarkdown} | 1 |
| {::nomarkdown}$(x^n_1, x^n_2, x^n_3)${:/nomarkdown} | {::nomarkdown}$(x^n_4, \cdots, x^n_{l_n})${:/nomarkdown} | 1 |
| $\vdots$ | $\vdots$ | $\vdots$ |
| {::nomarkdown}$(x^n_1, x^n_2, x^n_3, \cdots, x^n_{l_n-1})${:/nomarkdown} | {::nomarkdown}$(x^n_{l_n})${:/nomarkdown} | 1 |

and contributes once  for the following *token-level* input-output pairs

| {::nomarkdown}$\vec{\omega}_{\lt t} ${:/nomarkdown} | {::nomarkdown}$\omega_t${:/nomarkdown} | Count |
|---|---|---|
| $\emptyset$ | {::nomarkdown}$x^n_1${:/nomarkdown} | 1 |
| {::nomarkdown}$(x^n_1)${:/nomarkdown} | {::nomarkdown}$x^n_2${:/nomarkdown} | 1 |
| {::nomarkdown}$(x^n_1, x^n_2)${:/nomarkdown} | {::nomarkdown}$x^n_3${:/nomarkdown} | 1 |
| {::nomarkdown}$(x^n_1, x^n_2, x^n_3)${:/nomarkdown} | {::nomarkdown}$x^n_4${:/nomarkdown} | 1 |
| $\vdots$ | $\vdots$ | $\vdots$ |
| {::nomarkdown}$(x^n_1, x^n_2, x^n_3, \cdots, x^n_{l_n-1})${:/nomarkdown} | {::nomarkdown}$x^n_{l_n}${:/nomarkdown} | 1 |

> **Note**
>
> Note that we use superscript to denote the data sample index and use subscript to denote the token position index.

During pretraining, we drive the model to have as much world knowledge as possible via scaling $N \to \infty$, and we do not drive the model to solve specific language tasks $P(\vec{y}\mid\vec{x})$. Therefore, we only care about the whole sequence {::nomarkdown}$(x^n_1, x^n_2, \cdots, x^n_{l_n})${:/nomarkdown}. Or equivalently we only consider $P(\vec{y}\mid\vec{x})$  when $\vec{x}=\emptyset$ and {::nomarkdown}$\vec{y}=(x^n_1, x^n_2, \cdots, x^n_{l_n})${:/nomarkdown}.

Given this pretraining dataset $(\vec{x}^1, \vec{x}^2, \cdots, \vec{x}^N)$ (some of them may be identical and thus I do not write it as a set but a vector),
we can define another data-based empirical probability measure {::nomarkdown}$P_{\text{data}}${:/nomarkdown} over the sample space {::nomarkdown}$\Omega_{\text{data}}${:/nomarkdown} as follows,

$$
\begin{aligned}
& \Omega_{\text{data}} = \bigcup_{n=1}^N \lbrace \vec{x}^n \rbrace \subset \Omega_{\text{truth}}, \\
& P_{\text{data}} (\vec{\omega}) = \frac{ \sum_{n=1}^N \mathbb{1}_{ \{ \vec{x}^n = \vec{\omega} \}  } } {N}, \qquad \forall \vec{\omega} \in \Omega_{\text{data}},
\end{aligned}
$$

where {::nomarkdown}$\mathbb{1}_{ \{ \cdot \} }${:/nomarkdown} is the indicator function. Namely, we define the probability of a sequence as its **empirical counting frequency** in the pretrain dataset.

We can only train the network with parameter $\theta$ based on the pretraining dataset. We cannot access the ground-truth {::nomarkdown}$P_{\text{truth}}${:/nomarkdown}. Therefore,
it is very important that {::nomarkdown}$P_{\text{data}}${:/nomarkdown} is close to {::nomarkdown}$P_{\text{true}}${:/nomarkdown} in the sense that



$$
\begin{aligned}
& \Omega_{\text{data}} \to \Omega_{\text{truth}}, \\
& P_{\text{data}} (\vec{\omega}) \to P_{\text{truth}} (\vec{\omega}).
\end{aligned}
$$

Scaling laws [[9]](https://arxiv.org/abs/2001.08361) show that model quality improves as the pretraining corpus size $N$ grows.
It is very important that $N$ is large enough to cover the ground truth sample space as much as possible. In addition,
the quality of pretrain dataset also matters so that we can approximate the ground truth probability measure as much as possible.


Based on the previous analysis on NTP, we will make use of autoregressive factorization to model/learn {::nomarkdown}$P_{\text{data}}${:/nomarkdown}. Thus, we also need to model/learn the NTP-based token-level conditional probabilities {::nomarkdown}$P_{\text{data}, t}(\,\cdot \mid \vec{\omega}_{\lt t})${:/nomarkdown} from the pretrain dataset.

Specifically, we derive the data-based empirical token-level conditional probability {::nomarkdown}$P_{\text{data}, t}(\,\cdot \mid \vec{\omega}_{\lt t})${:/nomarkdown} as follows,

$$
\begin{aligned}
& N_{\text{data}, \vec{\omega}_{\lt t}}
= \sum_{n=1}^N \mathbb{1}_{ \left\{ t \le l_n, \text{ and the first } t-1 \text{ tokens of } \vec{x}^n \text{ is } \vec{\omega}_{\lt t} \right\} } \\
& N_{\text{data}, (\vec{\omega}_{\lt t}, \omega_t) }
= \sum_{n=1}^N \mathbb{1}_{ \left\{ t \le l_n, \text{ and the first } t \text{ tokens of } \vec{x}^n \text{ is } (\vec{\omega}_{\lt t}, \omega_t) \right\} } \\
& P_{\text{data}, t}(\vec{\omega}_{\lt t}, \omega_t)
= \frac{ N_{\text{data}, (\vec{\omega}_{\lt t}, \omega_t) } }{ \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \}}  }, \qquad \forall (\vec{\omega}_{\lt t}, \omega_t) \in \Omega_{\text{data}, t}  \\
& P_{\text{data}, t}(\,\omega_t \mid \vec{\omega}_{\lt t}) 
=
\begin{cases}
    \frac{ N_{\text{data}, (\vec{\omega}_{\lt t}, \omega_t) } } { N_{\text{data}, \vec{\omega}_{\lt t}} }; \text{ if }  N_{\text{data}, \vec{\omega}_{\lt t}} \gt 0 \\
    0; \text{ otherwise.}
\end{cases}
\end{aligned}
$$

Note that {::nomarkdown}$N_{\text{data}, \vec{\omega}_{\lt t}}${:/nomarkdown} is not the number of data samples that have prefix subsequence {::nomarkdown}$\vec{\omega}_{\lt t}${:/nomarkdown}, but the number of
data samples **of length at least $t$** that have prefix subsequence {::nomarkdown}$\vec{\omega}_{\lt t}${:/nomarkdown}.

Similarly to the definition of {::nomarkdown}$\Omega_{\text{truth}, t}${:/nomarkdown}, here {::nomarkdown}$\Omega_{\text{data}, t}${:/nomarkdown} is the set of all length-$t$ prefix subsequence of the pretraining dataset $(\vec{x}^1, \vec{x}^2, \cdots, \vec{x}^N)$. Also,  similar to  {::nomarkdown}$P_{\text{truth}, t}${:/nomarkdown}, {::nomarkdown}$P_{\text{data}, t}${:/nomarkdown} is the probability measure over {::nomarkdown}$\Omega_{\text{data}, t}${:/nomarkdown}. The conditional probability {::nomarkdown}$P_{\text{data}, t}(\,\cdot \mid \vec{\omega}_{\lt t})${:/nomarkdown}
can thus be derived from the probability measure {::nomarkdown}$P_{\text{data}, t}${:/nomarkdown}.

Similar to {::nomarkdown}$\Omega_{\text{truth}, \lt t}${:/nomarkdown}, we also define the sample space {::nomarkdown}$\Omega_{\text{data}, \lt t}${:/nomarkdown} as

$$
\begin{aligned}
\Omega_{\text{data}, \lt t} = \Big\lbrace
\vec{\omega}_{\lt t} = (\omega_1, \omega_2, \cdots, \omega_{t-1}): 
\vec{w}=(\vec{\omega}_{t-1}, \omega_t, \omega_{t+1}, \cdots, \omega_{l}) \in \Omega_{\text{data}}, l \ge t
\Big\rbrace.
\end{aligned}
$$

We can easily prove that {::nomarkdown}$P_{\text{data}}${:/nomarkdown} defined by sequence-level frequencies can be obtained by {::nomarkdown}$P_{\text{data}, t}(\,\cdot \mid \vec{\omega}_{\lt t}) \quad (1 \le t \le L_{\max})${:/nomarkdown} defined by token-level frequencies and vice verse. They are equivalent representations for the pretrain dataset.


### 3.2. The Fundamental Pretraining Objective: Forward KL Divergence for Mode Covering

In Sec. 2, our **sequence-level** goal is to find a single parameter vector $\theta$ such that the neural-network probability measure {::nomarkdown}$P_\theta${:/nomarkdown} over {::nomarkdown}$\Omega_{\theta}${:/nomarkdown} is as close as possible to the ground-truth measure {::nomarkdown}$P_{\text{truth}}${:/nomarkdown} over {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown}. However, we cannot access  {::nomarkdown}$P_{\text{truth}}${:/nomarkdown}, but instead we can only observe finitely many pretraining sequences $(\vec{x}^1,\ldots,\vec{x}^N)$ as shown in Sec. 3.1. Thus, we optimize against the empirical measure {::nomarkdown}$P_{\text{data}}${:/nomarkdown} on {::nomarkdown}$\Omega_{\text{data}}${:/nomarkdown}, with the scaling-law hope that {::nomarkdown}$P_{\text{data}} \to P_{\text{truth}}${:/nomarkdown} as $N \to \infty$.

The neural network with parameters $\theta$ produces predicted/estimated probability measure {::nomarkdown}$P_{\theta}${:/nomarkdown} over the sample space {::nomarkdown}$\Omega_{\theta}${:/nomarkdown}.
A neural network can generally take any input and thus {::nomarkdown}$\Omega_{\theta}${:/nomarkdown} could be larger than {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown} and thus {::nomarkdown}$\Omega_{\text{data}}${:/nomarkdown}, namely

$$
\Omega_{\text{data}} \subset \Omega_{\text{truth}} \subset \Omega_{\theta}
$$

Think about ChatGPT can take any prompt as input and then output any response. The resulting sequence could be not in any world knowledge in {::nomarkdown}$\Omega_{\text{data}}${:/nomarkdown}
or even not a valid language sequence in {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown}. If it produce a sequence out of {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown}, we say it results in **hallucination**.
We should try our best to train a LLM with parameter $\theta$ that produces valid sequences, i.e.,  {::nomarkdown}$\Omega_{\theta}${:/nomarkdown} should be as close to {::nomarkdown}$\Omega_{\text{truth}} ${:/nomarkdown} as possible.
Thus, the above equation is something like the squeeze theorem in Calculus. We aim at both the left {::nomarkdown}$\Omega_{\text{data}}${:/nomarkdown} and the right {::nomarkdown}$\Omega_{\theta}${:/nomarkdown} converge to {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown}.

Now we aim at {::nomarkdown}$P_{\theta}${:/nomarkdown} approximates {::nomarkdown}$P_{\text{data}}${:/nomarkdown}. We thus need a distance metric between such two **sequence-level** measures.  Kullback–Leibler (KL) divergence is the standard information-theoretic choice to evaluate  the distance between two probability distributions.
KL divergence is well defined for two distributions over the same sample space. Since {::nomarkdown}$\Omega_{\theta}${:/nomarkdown} could be larger than {::nomarkdown}$\Omega_{\text{data}}${:/nomarkdown},
we need to manipulate it. The natural solution is to extend {::nomarkdown}$\Omega_{\text{data}}${:/nomarkdown} to {::nomarkdown}$\Omega_{\theta}${:/nomarkdown} by defining

$$
P_{\text{data}} (\vec{\omega}) = 0,  \qquad \forall \vec{\omega} \in \Omega_{\theta} \setminus  \Omega_{\text{data}}.
$$

Now we can define KL divergence. However, we still has another problem: KL divergence has two directions. One is the forward KL divergence defined by

$$
\begin{aligned}
\mathrm{KL}\!\left(P_{\text{data}} \,\|\, P_\theta\right)
= \sum_{\vec{\omega}\in\Omega_{\theta}} P_{\text{data}}(\vec{\omega}) \log \frac{P_{\text{data}}(\vec{\omega})}{P_\theta(\vec{\omega})} 
= \sum_{\vec{\omega}\in\Omega_{\text{data}}} P_{\text{data}}(\vec{\omega}) \log \frac{P_{\text{data}}(\vec{\omega})}{P_\theta(\vec{\omega})}.
\end{aligned}
$$

The other is the reverse KL divergence defined by

$$
\begin{aligned}
\mathrm{KL}\!\left(P_\theta \,\|\, P_{\text{data}}\right) 
= \sum_{\vec{\omega}\in\Omega_{\theta}} P_\theta(\vec{\omega}) \log \frac{P_\theta(\vec{\omega})}{P_{\text{data}}(\vec{\omega})}.
\end{aligned}
$$

These two are not equal because KL divergence is not a symmetric distance. Which one should we use for pretraining? We use the **forward** direction {::nomarkdown}$\mathrm{KL}(P_{\text{data}}\|P_\theta)${:/nomarkdown} for the following two reasons:


1. **Mode-covering (zero-forcing).** If {::nomarkdown}$P_{\text{data}}(\vec{\omega}) \gt 0${:/nomarkdown} but {::nomarkdown}$P_\theta(\vec{\omega}) = 0${:/nomarkdown}, then {::nomarkdown}$\log\frac{P_{\text{data}}(\vec{\omega})}{P_\theta(\vec{\omega})} = +\infty${:/nomarkdown} and the objective is infinite. The optimizer for forward KL divergence is therefore pushed to assign **positive** probability to every sequence that appears in the pretraining corpus. This is what we want for pretraining: cover the world knowledge as much as possible, compress and explain them, and do not ignore rare but real patterns.
Reverse KL {::nomarkdown}$\mathrm{KL}(P_\theta\|P_{\text{data}})${:/nomarkdown} instead encourages **mode-seeking**: {::nomarkdown}$P_\theta${:/nomarkdown} may collapse onto a few high-probability modes and assign near-zero mass elsewhere, which is not pretraining goal. Instead, reverse KL is used during RL phase and I will explain it later. This mode-covering goal is the main reason to use forward KL during pretraining.

2. **We sample from data, not from the model.** Pretraining draws sequences from the pretraining corpus {::nomarkdown}$\Omega_{\text{data}}${:/nomarkdown}, not from {::nomarkdown}$\Omega_{\theta}${:/nomarkdown}.
Forward KL is an expectation under {::nomarkdown}$P_{\text{data}}${:/nomarkdown} (or, in the ideal limit, {::nomarkdown}$P_{\text{truth}}${:/nomarkdown}). Reverse KL is an expectation under {::nomarkdown}$P_{\theta}${:/nomarkdown}, which
is $+\infty$ when {::nomarkdown}$\Omega_{\theta}${:/nomarkdown} is strictly larger than {::nomarkdown}$\Omega_{\text{data}}${:/nomarkdown} (there exists a {::nomarkdown}$\vec{\omega} \in \Omega_{\theta} \setminus  \Omega_{\text{data}}${:/nomarkdown}).
This condition is easy to be satisfied and it is hard to prevent pretraining from satisfying this condition. Thus, reverse KL is not a good choice.

Now our pretraining problem becomes

$$
\boxed{\;\min_{\theta}\; \mathrm{KL}\!\left(P_{\text{data}} \,\|\, P_\theta\right).\;}
$$

In the ideal limit {::nomarkdown}$P_{\text{data}} \to P_{\text{truth}}${:/nomarkdown}, this is equivalent to {::nomarkdown}$\min_\theta \mathrm{KL}(P_{\text{truth}}\|P_\theta)${:/nomarkdown}. Next I will show how this single **sequence-level** objective become the **token-level** cross-entropy loss implemented in every GPT trainer.


### 3.3. From Sequence-Level KL to Token-Level KLs

We have already explained in Sec. 2.2 that the sequence-level measure is hard to model/compute due to the astronomically large sample space {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown} and we should use
NTP-based token-level measure.  For any {::nomarkdown}$\vec{\omega} = (\omega_1,\ldots,\omega_l) \in \Omega_{\text{data}}${:/nomarkdown}, we have

$$
\begin{aligned}
\log \frac{P_{\text{data}}(\vec{\omega})}{P_\theta(\vec{\omega})} 
= \log \frac{ \prod_{t=1}^l P_{\text{data}}(\omega_t \mid \vec{\omega}_{\lt t}) }{  \prod_{t=1}^l P_{\theta}(\omega_t \mid \vec{\omega}_{\lt t})  } 
= \sum_{t=1}^{l} \log \frac{P_{\text{data}}(\omega_t \mid \vec{\omega}_{\lt t})}{P_\theta(\omega_t \mid \vec{\omega}_{\lt t})}.
\end{aligned}
$$

Let {::nomarkdown}$\vec{X} = (X_1, X_2, \cdots, X_L)${:/nomarkdown} be a random sequence (of random length) drawn from {::nomarkdown}$ \Omega_{\text{data}}${:/nomarkdown} via the probability measure {::nomarkdown}$P_{\text{data}}${:/nomarkdown}. Then we get the forward KL divergence
in the following expectation format

$$
\begin{aligned}
\mathrm{KL}(P_{\text{data}}\|P_\theta)
&= \mathbb{E}_{\vec{X}\sim P_{\text{data}}}\!\left[\sum_{t=1}^{L} \log \frac{P_{\text{data}}(X_t \mid \vec{X}_{\lt t})}{P_\theta(X_t \mid \vec{X}_{\lt t})}\right] \\
&= \mathbb{E}_{\vec{X}\sim P_{\text{data}}}\!\left[\sum_{t=1}^{L_{\max}} \mathbb{1}_{\{t \le L\}} \log \frac{P_{\text{data}}(X_t \mid \vec{X}_{\lt t})}{P_\theta(X_t \mid \vec{X}_{\lt t})}\right],
\end{aligned}
$$

where $L$ is the (random) length of $\vec{X}$ and the sum runs over the tokens of that sequence and we take convention that
{::nomarkdown}$\log \frac{P_{\text{data}}(X_t \mid \vec{X}_{\lt t})}{P_\theta(X_t \mid \vec{X}_{\lt t})}=0${:/nomarkdown} when $t \gt L$. Exchanging the sum with expectation gives

$$
\begin{aligned}
\mathrm{KL}(P_{\text{data}}\|P_\theta)
&= \sum_{t = 1}^{L_{\max}} \mathbb{E}_{\vec{X}\sim P_{\text{data}}}\!\left[\mathbb{1}_{\{t \le L\}}\, \log \frac{P_{\text{data}}(X_t \mid \vec{X}_{\lt t})}{P_\theta(X_t \mid \vec{X}_{\lt t})}\right] \\
&= \sum_{t = 1}^{L_{\max}} \mathbb{E}_{ (\vec{X}_{\lt t}, X_t) \sim P_{\text{data},t}}\!\left[\log \frac{P_{\text{data}}(X_t \mid \vec{X}_{\lt t})}{P_\theta(X_t \mid \vec{X}_{\lt t})}\right].
\end{aligned}
$$

The inner equality holds because

$$
\begin{aligned}
&\mathbb{E}_{\vec{X}\sim P_{\text{data}}}\!\left[\mathbb{1}_{\{t \le L\}}\, \log \frac{P_{\text{data}}(X_t \mid \vec{X}_{\lt t})}{P_\theta(X_t \mid \vec{X}_{\lt t})}\right] \\
&=\sum_{\vec{\omega} \in \Omega_{\text{data}}} P_{\text{data}} (\vec{X} = \vec{\omega})
\left[\mathbb{1}_{\{t \le l\}}\, \log \frac{P_{\text{data}}(\omega_t \mid \vec{\omega}_{\lt t})}{P_\theta(\omega_t \mid \vec{\omega}_{\lt t})}\right] \\
&=\sum_{ (\vec{\omega}_{\lt t}, \omega_t) \in \Omega_{\text{data},t}}
\left[ \sum_{\substack{\vec{\nu} \in \Omega_{\text{data}}: t \le l,\, \vec{\nu}_{\lt t}=\vec{\omega}_{\lt t},\, \nu_t = \omega_t}}
P_{\text{data}} (\vec{X} = \vec{\nu})  \right]
\cdot \left[\log \frac{P_{\text{data}}(\omega_t \mid \vec{\omega}_{\lt t})}{P_\theta(\omega_t \mid \vec{\omega}_{\lt t})}\right] \\
&=\sum_{ (\vec{\omega}_{\lt t}, \omega_t) \in \Omega_{\text{data},t}}  P_{\text{data}} (\vec{\omega}_{\lt t}, \omega_t)
\left[\log \frac{P_{\text{data}}(\omega_t \mid \vec{\omega}_{\lt t})}{P_\theta(\omega_t \mid \vec{\omega}_{\lt t})}\right] \\
&= \mathbb{E}_{ (\vec{X}_{\lt t}, X_t) \sim P_{\text{data},t}}\!\left[\log \frac{P_{\text{data}}(X_t \mid \vec{X}_{\lt t})}{P_\theta(X_t \mid \vec{X}_{\lt t})}\right].
\end{aligned}
$$


By the law of total expectations conditioning on {::nomarkdown}$\vec{X}_{\lt t}=\vec{\omega}_{\lt t}${:/nomarkdown}, we then have

$$
\begin{aligned}
\mathrm{KL}(P_{\text{data}}\|P_\theta)
&= \sum_{t = 1}^{L_{\max}} \sum_{\vec{\omega}_{\lt t} \in \Omega_{\text{data}, {\lt t}}}
P_{\text{data}}(\vec{X}_{\lt t} = \vec{\omega}_{\lt t} ) 
 \cdot \mathbb{E}_{ (\vec{X}_{\lt t}, X_t) \sim P_{\text{data},t}}\!\left[
\left. \log \frac{P_{\text{data}}(X_t \mid \vec{X}_{\lt t})}{P_\theta(X_t \mid \vec{X}_{\lt t})}
\right|  \vec{X}_{\lt t} = \vec{\omega}_{\lt t} \right] \\
&= \sum_{t = 1}^{L_{\max}} \sum_{\vec{\omega}_{\lt t} \in \Omega_{\text{data}, {\lt t}} }
\left[ P_{\text{data}}(\vec{X}_{\lt t} = \vec{\omega}_{\lt t} ) \cdot
\mathrm{KL}\!\left(P_{\text{data}}(\cdot \mid \vec{\omega}_{\lt t}) \,\big\|\, P_\theta(\cdot \mid \vec{\omega}_{\lt t})\right)
\right] \\
&= \sum_{t = 1}^{L_{\max}} \mathbb{E}_{\vec{X}_{\lt t} \sim P_{\text{data},\lt t}}\!\left[
\mathrm{KL}\!\left(P_{\text{data}}(\cdot \mid \vec{X}_{\lt t}) \,\big\|\, P_\theta(\cdot \mid \vec{X}_{\lt t})\right)
\right],
\end{aligned}
$$

where {::nomarkdown}$\vec{X}_{\lt t} \sim P_{\text{data},\lt t}${:/nomarkdown} means that {::nomarkdown}$\vec{X}_{\lt t}${:/nomarkdown} follows the measure {::nomarkdown}$P_{\text{data},\lt t}${:/nomarkdown}, which is the probability distribution over all length-$(t-1)$ prefix subsequences of all data samples of length at least $t$. In addition, the inner equality holds because

$$
\begin{aligned}
&\mathbb{E}_{ (\vec{X}_{\lt t}, X_t) \sim P_{\text{data},t}}\!\left[ \left. \log \frac{P_{\text{data}}(X_t \mid \vec{X}_{\lt t})}{P_\theta(X_t \mid \vec{X}_{\lt t})} \right|  \vec{X}_{\lt t} = \vec{\omega}_{\lt t} \right] \\
&= \sum_{ (\vec{\nu}_{\lt t}, \nu_t) \in \Omega_{\text{data},t}} P_{\text{data}} \left( (\vec{X}_{\lt t}, X_t)=(\vec{\nu}_{\lt t}, \nu_t) \mid \vec{X}_{\lt t} = \vec{\omega}_{\lt t} \right)
\cdot \left[ \log \frac{P_{\text{data}}(\nu_t \mid \vec{\nu}_{\lt t})}{P_\theta(\nu_t \mid \vec{\nu}_{\lt t})} \right] \\
&= \sum_{ \nu_t \in \mathcal{V} \cup \{ \text{<bos>} \}} P_{\text{data}} \left( X_t=\nu_t \mid \vec{X}_{\lt t} = \vec{\omega}_{\lt t} \right)
\cdot \left[ \log \frac{P_{\text{data}}(\nu_t \mid \vec{\omega}_{\lt t})}{P_\theta(\nu_t \mid \vec{\omega}_{\lt t})} \right] \\
&= \sum_{ \nu_t \in \mathcal{V} \cup \{ \text{<bos>} \} } P_{\text{data}} (\nu_t \mid \vec{\omega}_{\lt t})
\cdot \left[ \log \frac{P_{\text{data}}(\nu_t \mid \vec{\omega}_{\lt t})}{P_\theta(\nu_t \mid \vec{\omega}_{\lt t})} \right] \\
&=\mathrm{KL}\!\left(P_{\text{data}}(\cdot \mid \vec{\omega}_{\lt t}) \,\big\|\, P_\theta(\cdot \mid \vec{\omega}_{\lt t})\right).
\end{aligned}
$$


Now we have reached the milestone that **the sequence-level KL is the sum of the token-level expected KLs between conditional distributions**, where the expectation at token position $t$ is taken over prefix subsequence before the current token position $t$. This echoes my previous question about how to optimize  $\theta$ across positions. We minimize one global divergence via summation, not {::nomarkdown}$L_{\max}${:/nomarkdown} independent ones.

Furthermore, a prefix subsequence {::nomarkdown}$\vec{\omega}_{\lt t}${:/nomarkdown} that never appears in the training set has {::nomarkdown}$N_{\text{data}, \vec{\omega}_{\lt t}} = 0${:/nomarkdown} and does not enter the expected KL at position $t$ (there is no data mass on that prefix). Prefixes that appear many times contribute proportionally more to the expectation. We can make it more explictly.



$$
\begin{aligned}
\mathrm{KL}(P_{\text{data}}\|P_\theta)
&= \sum_{t = 1}^{L_{\max}} \sum_{\vec{\omega}_{\lt t} \in \Omega_{\text{data}, \lt t}}
\left[ P_{\text{data}}(\vec{X}_{\lt t} = \vec{\omega}_{\lt t} )
\cdot \mathrm{KL}\!\left(P_{\text{data}}(\cdot \mid \vec{\omega}_{\lt t}) \,\big\|\, P_\theta(\cdot \mid \vec{\omega}_{\lt t})\right) \right] \\
&= \sum_{t = 1}^{L_{\max}} \sum_{\vec{\omega}_{\lt t} \in \Omega_{\text{data}, \lt t}}
\left[ \frac{ N_{ \text{data}, \vec{\omega}_{\lt t} } } { \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }}
\cdot \mathrm{KL}\!\left(P_{\text{data}}(\cdot \mid \vec{\omega}_{\lt t}) \,\big\|\, P_\theta(\cdot \mid \vec{\omega}_{\lt t})\right) \right] \\
&= \sum_{t = 1}^{L_{\max}} \left\{ \frac{1}{ \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }}
\sum_{\vec{\omega}_{\lt t} \in \Omega_{\text{data}, \lt t}}
\left[ N_{\text{data}, \vec{\omega}_{\lt t} }
\cdot \mathrm{KL}\!\left(P_{\text{data}}(\cdot \mid \vec{\omega}_{\lt t}) \,\big\|\, P_\theta(\cdot \mid \vec{\omega}_{\lt t})\right) \right] \right\} \\
&= \sum_{t = 1}^{L_{\max}} \left\{ \frac{1}{ \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }}
\sum_{n=1}^N \left[ \mathbb{1}_{ \{ t \le l_n \} }
\cdot \mathrm{KL}\!\left(P_{\text{data}}(\cdot \mid \vec{x}^n_{\lt t}) \,\big\|\, P_\theta(\cdot \mid \vec{x}^n_{\lt t})\right) \right] \right\} \\
&= \sum_{t = 1}^{L_{\max}} \left[ \frac{ \sum_{n=1}^N  \left[ \mathbb{1}_{ \{ t \le l_n \} } \cdot \mathrm{KL}\!\left(P_{\text{data}}(\cdot \mid \vec{x}^n_{\lt t}) \,\big\|\, P_\theta(\cdot \mid \vec{x}^n_{\lt t})\right) \right]  }{ \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }}  \right].
\end{aligned}
$$

For each position $t$, we average the token-level KL over all training sequences that actually have a token at that position, then sum those averages over $t$. Equivalently, sequence $n$ enters position $t$ iff {::nomarkdown}$t \le l_n${:/nomarkdown} (the same indicator {::nomarkdown}$\mathbb{1}_{\{t \le L\}}${:/nomarkdown} used when exchanging the sum with expectation above).

Let us consider a toy example with $N=5$, {::nomarkdown}$L_{\max}=4${:/nomarkdown}. Namely, the pretrain dataset is $(\vec{x}^1,\vec{x}^2,\vec{x}^3,\vec{x}^4,\vec{x}^5)$.
Suppose the five data samples have lengths {::nomarkdown}$l_1=2${:/nomarkdown}, {::nomarkdown}$l_2=3${:/nomarkdown}, {::nomarkdown}$l_3=3${:/nomarkdown}, {::nomarkdown}$l_4=4${:/nomarkdown}, and {::nomarkdown}$l_5=4${:/nomarkdown}, respectively. We denote

$$
f(n,t) \;=\; \mathrm{KL}\!\left(P_{\text{data}}(\cdot \mid \vec{x}^n_{\lt t}) \,\big\|\, P_\theta(\cdot \mid \vec{x}^n_{\lt t})\right).
$$

Each filled cell below is one term $f(n,t)$; empty cells mean sequence $n$ is too short at position $t$.

```
              t = 1     t = 2       t = 3      t = 4
            ---------  ---------  ---------  ---------
 x^1, l=2  | f(1,1)   | f(1,2)   |          |          |
 x^2, l=3  | f(2,1)   | f(2,2)   | f(2,3)   |          |
 x^3, l=3  | f(3,1)   | f(3,2)   | f(3,3)   |          |
 x^4, l=4  | f(4,1)   | f(4,2)   | f(4,3)   | f(4,4)   |
 x^5, l=4  | f(5,1)   | f(5,2)   | f(5,3)   | f(5,4)   |
```

The last line of the derivation is then "sum the column, divide by the number of filled cells, repeat for each $t$":

$$
\begin{aligned}
\mathrm{KL}(P_{\text{data}}\|P_\theta) 
& = \underbrace{\frac{f(1,1)+f(2,1)+f(3,1)+f(4,1)+f(5,1)}{5}}_{t=1} \\
& + \underbrace{\frac{f(1,2)+f(2,2)+f(3,2)+f(4,2)+f(5,2)}{5}}_{t=2} \\
& + \underbrace{\frac{f(2,3)+f(3,3)+f(4,3)+f(5,3)}{4}}_{t=3} \\
& + \underbrace{\frac{f(4,4)+f(5,4)}{2}}_{t=4}.
\end{aligned}
$$

We do **not** add $5 \times 4 = 20$ terms and treat missing cells as zero; shorter sequences are excluded from the average at later $t$, which is why the denominators drop from $5$ to $4$ to $2$. Duplicate prefixes {::nomarkdown}$\vec{\omega}_{\lt t}${:/nomarkdown} shared by several rows are still handled correctly in the prefix-count form above: they appear as repeated $f(n,t)$ values with the same KL inside a column average.

Now we have equivalently transferred the sequence-level KL into the summation of token-level KLs. Hence, our pretrain problem becomes

$$
\boxed{
\begin{aligned}
\min_{\theta}\; \mathrm{KL}\!\left(P_{\text{data}} \,\|\, P_\theta\right)
&= \min_{\theta}\; \sum_{t = 1}^{L_{\max}} \left[ \frac{ \sum_{n=1}^N  \left[ \mathbb{1}_{ \{ t \le l_n \} } \cdot \mathrm{KL}\!\left(P_{\text{data}}(\cdot \mid \vec{x}^n_{\lt t}) \,\big\|\, P_\theta(\cdot \mid \vec{x}^n_{\lt t})\right) \right]  }{ \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }}  \right] .
\end{aligned}
}
$$


### 3.4. From Token-Level KL to Token-Level Cross-Entropy Loss

For two distributions $p$ and $q$ over the same sample space $\mathcal{X}$, we have

$$
\begin{aligned}
\mathrm{KL}(p \,\|\, q) 
& = \mathbb{E}_{X \sim p}\!\left[\log \frac{p(X)}{q(X)}\right] 
= \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)} \\
& = \sum_{x \in \mathcal{X}} p(x) \log p(x) - \sum_{x \in \mathcal{X}} p(x) \log q(x) \\
& = - \left[ -\sum_{x \in \mathcal{X}} p(x) \log p(x) \right] + \left[- \sum_{x \in \mathcal{X}} p(x) \log q(x) \right] \\
& = - H(p) + H(p,q).
\end{aligned}
$$

Thus,

$$
H(p,q) = H(p) + \mathrm{KL}(p \,\|\, q).
$$

In particular, there three terms have the following meanings.


| Term                        | Name           | Meaning                                                                                  |
| --------------------------- | -------------  | -----------------------------------------------------------------------------------------|
| $H(p)$                      | (True) Entropy | Average bits/nats needed if we use the *optimal* code for $p$ to represent $p$           |
| $\mathrm{KL}(p \,\Vert\, q)$ | KL Divergence  | Extra bits/nats (*penalty*) needed if we use the *optimal* code for $q$ to represent $p$ |
| $H(p,q)$                    | Cross-Entropy  | Average bits/nats needed if we use the *optimal* code for $q$ to represent $p$           |


For our studied token-level KL divergence, we thus have

$$
\begin{aligned}
\mathrm{KL}\!\left(P_{\text{data}}(\cdot \mid \vec{x}^n_{\lt t}) \,\big\|\, P_\theta(\cdot \mid \vec{x}^n_{\lt t})\right) 
= -H(P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t})) + H(P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t}),\,P_\theta(\cdot\mid \vec{x}^n_{\lt t}))
\end{aligned}
$$

The (true) entropy {::nomarkdown}$H(P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t}))${:/nomarkdown} depends only on the data, **not on $\theta$**. Therefore,
minimize the token-level KL divergence is equivalent to minimize the token-level cross-entropy,

$$
\begin{aligned}
\arg\min_\theta \mathrm{KL}\!\left(P_{\text{data}}(\cdot \mid \vec{x}^n_{\lt t}) \,\big\|\, P_\theta(\cdot \mid \vec{x}^n_{\lt t})\right) 
= \arg\min_\theta H(P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t}),\,P_\theta(\cdot\mid \vec{x}^n_{\lt t})).
\end{aligned}
$$


For the global sequence-level KL divergence, we have

$$
\begin{aligned}
\mathrm{KL}\!\left(P_{\text{data}} \,\|\, P_\theta\right)
&= \sum_{t = 1}^{L_{\max}}
\frac{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
  \mathrm{KL}\!\left(P_{\text{data}}(\cdot \mid \vec{x}^n_{\lt t}) \,\big\|\, P_\theta(\cdot \mid \vec{x}^n_{\lt t})\right)
}{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
} \\
&= - \sum_{t = 1}^{L_{\max}}
\frac{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
  H(P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t}))
}{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
} \\
&\quad + \sum_{t = 1}^{L_{\max}}
\frac{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
  H(P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t}),\,P_\theta(\cdot\mid \vec{x}^n_{\lt t}))
}{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
}.
\end{aligned}
$$

We can go through the same derivation in Sec. 3.3 to get the sequence-level entropy/cross-entropy to token-Level entropies/cross-entropies. Namely, the sequence-level (true) entropy of {::nomarkdown}$P_{\text{data}}${:/nomarkdown} is

$$
\begin{aligned}
H(P_{\text{data}})
&= \sum_{t = 1}^{L_{\max}}
\frac{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
  H(P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t}))
}{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
}.
\end{aligned}
$$

and the sequence-level cross entropy between {::nomarkdown}$P_{\text{data}}${:/nomarkdown} and {::nomarkdown}$P_{\theta}${:/nomarkdown} is

$$
\begin{aligned}
H(P_{\text{data}}, P_{\theta})
&= \sum_{t = 1}^{L_{\max}}
\frac{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
  H(P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t}),\,P_\theta(\cdot\mid \vec{x}^n_{\lt t}))
}{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
}.
\end{aligned}
$$

Of course, we have

$$
H(P_{\text{data}}, P_{\theta}) = H(P_{\text{data}}) + \mathrm{KL}\!\left(P_{\text{data}} \,\|\, P_\theta\right),
$$

and I would like to emphasize the physical meaning of these three terms again

| Term                        | Name           | Meaning                                                                                  |
| --------------------------- | -------------  | -----------------------------------------------------------------------------------------|
| {::nomarkdown}$H(P_{\text{data}})${:/nomarkdown}                                          | Empirical Corpus Entropy  | Average bits/nats needed if we use the *optimal* code for pretrain sequence-level distribution {::nomarkdown}$P_{\text{data}}${:/nomarkdown} to represent {::nomarkdown}$P_{\text{data}}${:/nomarkdown}           |
| {::nomarkdown}$\mathrm{KL}\!\left(P_{\text{data}} \,\Vert\, P_\theta\right)${:/nomarkdown} | Forward KL Divergence     | Extra bits/nats (*penalty*) needed if we use the *optimal* code for  predicted/estimated sequence-level distribution {::nomarkdown}$P_{\theta}${:/nomarkdown} to represent {::nomarkdown}$P_{\text{data}}${:/nomarkdown}  |
| {::nomarkdown}$H(P_{\text{data}}, P_{\theta})${:/nomarkdown}                              | Cross-Entropy             | Average bits/nats needed if we use the *optimal* code for predicted/estimated sequence-level distribution {::nomarkdown}$P_{\theta}${:/nomarkdown} to represent {::nomarkdown}$P_{\text{data}}${:/nomarkdown}         |

Therefore, minimizing sequence-level KL divergence is equivalent to minimizing sequence-level cross-entropy, yielding
same optimal parameter $\theta^*$ and a constant gap on the optimal values, i.e.,

$$
\begin{aligned}
& \theta^* = \arg\min_\theta \mathrm{KL}\!\left(P_{\text{data}} \,\|\, P_\theta\right) 
= \arg\min_\theta H(P_{\text{data}}, P_{\theta}); \\
& H(P_{\text{data}}, P_{\theta^*}) = H(P_{\text{data}}) + \mathrm{KL}\!\left(P_{\text{data}} \,\|\, P_\theta^*\right),
\end{aligned}
$$

Therefore, our pretrain problem now becomes

$$
\boxed{
\begin{aligned}
\min_{\theta}\; H(P_{\text{data}}, P_{\theta})
&= \min_{\theta}\; \sum_{t = 1}^{L_{\max}}
\frac{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
  H(P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t}),\,P_\theta(\cdot\mid \vec{x}^n_{\lt t}))
}{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
}.
\end{aligned}
}
$$


Note that the token-level cross entropy between conditional probabilities {::nomarkdown}$H(P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t}),\,P_\theta(\cdot\mid \vec{x}^n_{\lt t})) ${:/nomarkdown} cannot be computed by a single data sample,
but need to be computed based on all data samples with the same prefix subsequence {::nomarkdown}$\vec{x}^n_{\lt t}${:/nomarkdown}.
It is possible that   {::nomarkdown}$\vec{x}^n_{\lt t} = \vec{x}^{\tilde{n}}_{\lt t}${:/nomarkdown} for some $\tilde{n} \neq n$, and thus both $\vec{x}^n$ and $\vec{x}^{\tilde{n}}$ will contribute to the conditional probability {::nomarkdown}$P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t})${:/nomarkdown}. This makes the cross-entropy computation difficult because of the coupling between data samples. We seek to compute the individual contribution to the cross-entropy for any individual data sample **simply based on itself**. Towards that end, let us rephrase the position-$t$ total cross-entropy as follows,

$$
\begin{aligned}
& \sum_{n=1}^N  \left[ \mathbb{1}_{ \{ t \le l_n \} } \cdot H(P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t}),\,P_\theta(\cdot\mid \vec{x}^n_{\lt t})) \right] \\
&= \sum_{\vec{\omega}_{\lt t} \in \Omega_{\text{data}, \lt t}} \left[ \mathbb{1}_{ \{ t \le l_n \} } \cdot N_{\text{data}, \vec{\omega}_{\lt t}} \cdot H(P_{\text{data}}(\cdot\mid \vec{\omega}_{\lt t}),\,P_\theta(\cdot\mid \vec{\omega}_{\lt t})) \right].
\end{aligned}
$$

This holds because we simply combine same prefix subsequences together and also because {::nomarkdown}$N_{\text{data}, \vec{\omega}_{\lt t}}${:/nomarkdown} is the number of data samples of length at least $t$ that have prefix subsequence {::nomarkdown}$\vec{\omega}_{\lt t}${:/nomarkdown}. Note that

$$
\sum_{ \vec{\omega}_{\lt t} \in \Omega_{\text{data}, \lt t}} N_{\text{data}, \vec{\omega}_{\lt t}} =  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }.
$$


Now we have

$$
\begin{aligned}
& \sum_{n=1}^N  \left[ \mathbb{1}_{ \{ t \le l_n \} } \cdot H(P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t}),\,P_\theta(\cdot\mid \vec{x}^n_{\lt t})) \right] \\
& = \sum_{\vec{\omega}_{\lt t} \in \Omega_{\text{data}, \lt t}} \left[ \mathbb{1}_{ \{ t \le l_n \} } \cdot N_{\text{data}, \vec{\omega}_{\lt t}} \cdot H(P_{\text{data}}(\cdot\mid \vec{\omega}_{\lt t}),\,P_\theta(\cdot\mid \vec{\omega}_{\lt t})) \right] \\
&= - \sum_{\vec{\omega}_{\lt t} \in \Omega_{\text{data}, \lt t}} \left[ \mathbb{1}_{ \{ t \le l_n \} } \cdot N_{\text{data}, \vec{\omega}_{\lt t}} \cdot \sum_{ \omega_t \in \mathcal{V} \cup \{ \text{<bos>} \} } P_{\text{data}}( \omega_t \mid  \vec{\omega}_{\lt t}) \cdot \log P_\theta( \omega_t \mid \vec{\omega}_{\lt t}) \right] \\
&= - \sum_{\vec{\omega}_{\lt t} \in \Omega_{\text{data}, \lt t}} \left[ \mathbb{1}_{ \{ t \le l_n \} } \cdot \sum_{ \omega_t \in \mathcal{V} \cup \{ \text{<bos>} \} } N_{\text{data}, (\vec{\omega}_{\lt t}, \omega_t)}  \cdot \log P_\theta( \omega_t \mid \vec{\omega}_{\lt t}) \right] \\
&= \sum_{n=1}^N \left[ \mathbb{1}_{ \{ t \le l_n \} } \cdot \left( - \log P_\theta( x^n_t \mid \vec{x}^n_{\lt t}) \right) \right]
\end{aligned}
$$

where the last equality is due to counting for all {::nomarkdown}$(\vec{\omega}_{\lt t}, \omega_t)${:/nomarkdown} in the pretrain dataset. This is an important result because we have
reduced the token-level **total (not average!)** cross-entropy loss into individual data samples. For any given data sample $\vec{x}^n$, it contributions to the position-$t$ total cross-entropy loss by the amount

$$
\mathbb{1}_{ \{ t \le l_n \} } \cdot \left( - \log P_\theta( x^n_t \mid \vec{x}^n_{\lt t}) \right).
$$

Namely, if its length {::nomarkdown}$l_n < t${:/nomarkdown}, $\vec{x}^n$ does not contribute to the position-$t$ total cross-entropy loss; otherwise, it contributes  {::nomarkdown}$\left( - \log P_\theta( x^n_t \mid \vec{x}^n_{\lt t}) \right)${:/nomarkdown}, which is the negative log likelihood (NLL) loss of **this data sample itself**.

Now we conclude that our pretrain problem becomes

$$
\boxed{
\begin{aligned}
\min_{\theta}\; H(P_{\text{data}}, P_{\theta})
&= \min_{\theta}\; \sum_{t = 1}^{L_{\max}}
\frac{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
  H(P_{\text{data}}(\cdot\mid \vec{x}^n_{\lt t}),\,P_\theta(\cdot\mid \vec{x}^n_{\lt t}))
}{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
} \\
&= \min_{\theta}\; \sum_{t = 1}^{L_{\max}}
\frac{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
  \left( - \log P_\theta( x^n_t \mid \vec{x}^n_{\lt t}) \right)
}{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
}.
\end{aligned}
}
$$

This illustrates that the sequence-level cross-entropy loss is equal to the summation of **average** token-level cross-entropy loss over all token positions, and
is further equal to  the summation of **average** token-level NLL loss over all token positions.



Consider the previous toy example with $N=5$, {::nomarkdown}$L_{\max}=4${:/nomarkdown} again. We denote

$$
\mathrm{NLL}(n,t) \;=\; - \log P_\theta( x^n_t \mid \vec{x}^n_{\lt t}), \qquad 1 \le t \le l_n,
$$

and we depict them in the following table

```
           |   t = 1    |   t = 2    |  t = 3     |   t = 4    |
           | ---------- | ---------  |---------   | ---------  |
 x^1, l=2  | NLL(1,1)   | NLL(1,2)   |            |            |
 x^2, l=3  | NLL(2,1)   | NLL(2,2)   | NLL(2,3)   |            |
 x^3, l=3  | NLL(3,1)   | NLL(3,2)   | NLL(3,3)   |            |
 x^4, l=4  | NLL(4,1)   | NLL(4,2)   | NLL(4,3)   | NLL(4,4)   |
 x^5, l=4  | NLL(5,1)   | NLL(5,2)   | NLL(5,3)   | NLL(5,4)   |
```

Then we can get the sequence-level cross-entropy loss as

$$
\begin{aligned}
H(P_{\text{data}}, P_{\theta}) 
& = \underbrace{\frac{\mathrm{NLL}(1,1)+\mathrm{NLL}(2,1)+\mathrm{NLL}(3,1)+\mathrm{NLL}(4,1)+\mathrm{NLL}(5,1)}{5}}_{t=1} \\
& + \underbrace{\frac{\mathrm{NLL}(1,2)+\mathrm{NLL}(2,2)+\mathrm{NLL}(3,2)+\mathrm{NLL}(4,2)+\mathrm{NLL}(5,2)}{5}}_{t=2} \\
& + \underbrace{\frac{\mathrm{NLL}(2,3)+\mathrm{NLL}(3,3)+\mathrm{NLL}(4,3)+\mathrm{NLL}(5,3)}{4}}_{t=3} \\
& + \underbrace{\frac{\mathrm{NLL}(4,4)+\mathrm{NLL}(5,4)}{2}}_{t=4}.
\end{aligned}
$$

### 3.5 Assumption for Equal-length Pretraining Sequences

We will now show some assumptions to approximate the pretrain optimization object mainly for efficient implementation in practice.
In practice, we generally have

$$
N \gg L_{\max},
$$

i.e., the number of data samples (in the order of trillions of tokens) is much larger than the context window (less than 1M).
For example,  DeepSeek V4 [[10]](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) has {::nomarkdown}$L_{\max}=1M${:/nomarkdown}
and pretrained more than 32T tokens. In addition, the context window is generally progressively extended in stages. For example,  DeepSeek V4 pretrains data from the sequence length of 4K to 16K, 64K, and 1M. Suppose that we have 10T tokens
of length up to 4K for pretraining. Then we have

$$
\begin{aligned}
N
&\ge \frac{10T}{4K}
= \frac{10 \times 10^{12}}{4096}
= 2.44 \times 10^9 \\
&\approx 2.5B \gg L_{\max} = 4096 = 4K.
\end{aligned}
$$

This confirms that $N$ is much larger than {::nomarkdown}$L_{\max}${:/nomarkdown}. For each position $t$, if we want to get a forward pass or a backward pass
for the average token-level NLL loss, i.e.,

$$
\frac{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
  \left( - \log P_\theta( x^n_t \mid \vec{x}^n_{\lt t}) \right)
}{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
},
$$

we need to load all $N$ data samples, which is expensive. It is even worse that we need to
again load all $N$ data samples for compute another position's average token-level NLL loss. Namely,
we need to load data samples {::nomarkdown}$L_{\max}${:/nomarkdown} times into GPU for a single forward/backward pass.
Even if we use SGD to only load a mini-batch of data samples to approximate
the gradient, we still load this mini-batch {::nomarkdown}$L_{\max}${:/nomarkdown} times for a single forward/backward pass.
This is inefficient under {::nomarkdown}$N \gg L_{\max}${:/nomarkdown}. We hope that during each forward/backward pass,
we only need to load data samples once.

To achieve this goal, we will approximate {::nomarkdown}$\sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }${:/nomarkdown} by $N$, i.e.,


$$
\boxed{
\begin{aligned}
\min_{\theta}\; H(P_{\text{data}}, P_{\theta})
&= \min_{\theta}\; \sum_{t = 1}^{L_{\max}}
\frac{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
  \left( - \log P_\theta( x^n_t \mid \vec{x}^n_{\lt t}) \right)
}{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
} \\
&\approx  \min_{\theta}\; \sum_{t = 1}^{L_{\max}}
\frac{
  \sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }
  \left( - \log P_\theta( x^n_t \mid \vec{x}^n_{\lt t}) \right)
}{N} \\
&=   \min_{\theta}\; \frac{1}{N}  \sum_{n=1}^N
\sum_{t = 1}^{l_n}
\left( - \log P_\theta( x^n_t \mid \vec{x}^n_{\lt t}) \right).
\end{aligned}
}
$$


Under this approximation, we can exchange the summation over token position $t$ and the summation over data sample index $n$.
Now when we load any data sample $\vec{x}^n$, we can compute the summation of its NLL loss for all {::nomarkdown}$l_n \le L_{\max}${:/nomarkdown} token positions.
Namely, we can compute **data-sample-level** cross-entropy (NLL) loss and then compute average over all data samples.
We only need to load once for this data sample, and finish the computation at most {::nomarkdown}$L_{\max}${:/nomarkdown} times. It becomes much more I/O-efficient and
memory-efficient now!

How can we approximate {::nomarkdown}$\sum_{n=1}^N \mathbb{1}_{ \{ t \le l_n \} }${:/nomarkdown} by $N$? It simply means that
all $N$ data samples of length {::nomarkdown}$L_{\max}${:/nomarkdown}, i.e.,

$$
l_n = L_{\max}, \qquad \forall n \in \{1, 2, \cdots, N\}.
$$


This is the equal-length assumption. In practice, we either pad or pack sequences into equal lengths of {::nomarkdown}$L_{\max}${:/nomarkdown}.

Padding means that we add some special padding tokens until its length reaches {::nomarkdown}$L_{\max}${:/nomarkdown}, i.e.,

```
<bos> doc1 <bos> <pad> ... <pad> <bos>.
```

Packing means that we pack two or more short sequences into a long sequence until its length reaches {::nomarkdown}$L_{\max}${:/nomarkdown}, i.e.,

```
<bos> doc1 <bos> <doc2> ... <bos>.
```

Padding does not introduce cross-document information leakage but it wastes FLOPs. We can improve the implementation such that
NLL loss of padding tokens will not be computed and thus will not contribute to data-sample-level cross-entropy loss.
In this way, we can reduce the wasted FLOPs. Packing does not waste FLOPs but results in cross-document information leakage.
That is to say, `doc2` will attend to `doc1`, i.e.,  token prediction for `doc2` will depend on tokens of `doc1`.
Of course, since `<bos>` serves as break between sequences, such leakage should be of small impact.

Here we only talk about the short documents. In practice, there are also long documents of length larger than {::nomarkdown}$L_{\max}${:/nomarkdown}.
For such documents, we generally chunk them into sequences of length {::nomarkdown}$L_{\max}${:/nomarkdown} while leaving the tail sequence as a short sequence.
The tail sequence will either be padded or packed into a sequence of length {::nomarkdown}$L_{\max}${:/nomarkdown}.

However, in our `nanochat-ascend` project, Karpathy uses a different approach, which is crop-based best-fit packing algorithm.
As shown in [`nanochat/dataloader.py`](https://github.com/leideng/nanochat-ascend/blob/main/nanochat/dataloader.py), it works as follows
- Every row starts with BOS token
- Documents packed using best-fit algorithm to minimize cropping
- When no document fits remaining space, crops a document to fill exactly
- 100% utilization (no padding), ~35% tokens cropped at T=2048

In particular,  we compare this approach with standard concatenate-then-chunk approach as follows.

```text
Concatenate-then-chunk (standard):          BOS-aligned bestfit (this code):
─────────────────────────────────           ─────────────────────────────────
<bos> doc1... ...    <bos> doc2...          <bos> docX ... <bos> docY_cropped
...doc2_continued... <bos> doc3..           <bos> docZ ... <bos> docW_cropped
...doc3_continued... ..... <bos>            <bos> docA ...

✗ rows can start mid-document               ✓ EVERY row starts with <BOS>
✓ no tokens wasted                          ✗ ~35% tokens discarded (crops)
✗ tokens attend to prior-doc context        ✓ full context visible from BOS
```


In summary, with this equal-length preprocessing assumption/approximation, we have the following benefits in practical implementation:

1. **Data loading.** The data pipeline naturally iterates over $n$ (sequences): read one sequence of {::nomarkdown}$L_{\max}${:/nomarkdown} tokens, run one forward/backward pass, accumulate the per-sequence NLL {::nomarkdown}$\sum_{t=1}^{L} \mathrm{NLL}(n,t)${:/nomarkdown}, and move on. Instead, an outer loop over $t$ would require, for each position, a pass over all $N$ sequences—{::nomarkdown}$L_{\max}${:/nomarkdown} full corpus scans per epoch instead of one. When $N \sim 10^9$ and {::nomarkdown}$L_{\max} \sim 10^3${:/nomarkdown}, that represents orders-of-magnitude savings in I/O and data movement.

2. **Fixed-shape batches.** Equal-length sequences yield dense {::nomarkdown}$N_{\text{batch}} \times L_{\max}${:/nomarkdown} token tensors with no ragged lengths, no per-row padding waste, and no position-dependent mask logic in the loss. Loaders can use fixed byte offsets and memory-mapped reads; this alone can **significantly reduce data-loading time** compared to variable-length documents with dynamic padding.

3. **Alignment with autoregressive models.** A causal transformer already computes all {::nomarkdown}$L_{\max}${:/nomarkdown} next-token logits in one forward pass over a sequence. Summing $\mathrm{NLL}(n,t)$ over $t$ for fixed $n$ matches the natural compute graph, while summing over $n$ for fixed $t$ does not.



> **Note**
>
> Equal-length chunking/padding/packing is a **preprocessing** choice, not a claim that natural language documents (or the ground-truth sample space {::nomarkdown}$\Omega_{\text{truth}}${:/nomarkdown}) have fixed length. It seems that {::nomarkdown}$P_{\text{data}}${:/nomarkdown} deviates {::nomarkdown}$P_{\text{truth}}${:/nomarkdown} significantly. Based on our theory, a padded/packed sequence is even NOT a valid sequence! However, they are still close as long as $N$ is large enough. First, chunking long documents does not affect {::nomarkdown}$P_{\text{truth}}${:/nomarkdown} because both {::nomarkdown}$P_{\text{truth}}${:/nomarkdown}  and {::nomarkdown}$P_{\text{data}}${:/nomarkdown}  can only handle {::nomarkdown}$L_{\max}${:/nomarkdown} tokens at most. Second, padding/packing will insert `<bos>` between tokens. When we generate a response, it terminates when reaching `<bos>`. Thus, we still model short sequences. As long as almost all short sequences can be put into the beginning of some sequences in the pretrain dataset, {::nomarkdown}$P_{\text{text}}${:/nomarkdown} still approximates {::nomarkdown}$P_{\text{truth}}${:/nomarkdown}. This is some justification for the practical preprocessing. But I do think that here we need to be more careful in the sense that we can have room to more elegantly improve this preprocessing. I need some time to think about it further and perhaps I will discuss it some days later.




We again consider the previous toy example with $N=5$, {::nomarkdown}$L_{\max}=4${:/nomarkdown}. With equal-length assumption, we can depict them in the following table

```
           |   t = 1    |   t = 2    |  t = 3                |   t = 4             |
           | ---------- | ---------  |-------------------    | ------------------- |
 x^1, l=2  | NLL(1,1)   | NLL(1,2)   | NLL(1,3) (pad/pack)   | NLL(1,4) (pad/pack) |
 x^2, l=3  | NLL(2,1)   | NLL(2,2)   | NLL(2,3)              | NLL(2,4) (pad/pack) |
 x^3, l=3  | NLL(3,1)   | NLL(3,2)   | NLL(3,3)              | NLL(3,4) (pad/pack) |
 x^4, l=4  | NLL(4,1)   | NLL(4,2)   | NLL(4,3)              | NLL(4,4)            |
 x^5, l=4  | NLL(5,1)   | NLL(5,2)   | NLL(5,3)              | NLL(5,4)            |
```

Then we can get the approximate sequence-level cross-entropy loss as

$$
\begin{aligned}
H(P_{\text{data}}, P_{\theta}) \\
\approx \underbrace{\frac{\mathrm{NLL}(1,1)+\mathrm{NLL}(2,1)+\mathrm{NLL}(3,1)+\mathrm{NLL}(4,1)+\mathrm{NLL}(5,1)}{5}}_{t=1} \\
+ \underbrace{\frac{\mathrm{NLL}(1,2)+\mathrm{NLL}(2,2)+\mathrm{NLL}(3,2)+\mathrm{NLL}(4,2)+\mathrm{NLL}(5,2)}{5}}_{t=2} \\
+ \underbrace{\frac{\mathrm{NLL}(1,3)+\mathrm{NLL}(2,3)+\mathrm{NLL}(3,3)+\mathrm{NLL}(4,3)+\mathrm{NLL}(5,3)}{5}}_{t=3} \\
+ \underbrace{\frac{\mathrm{NLL}(1,4)+\mathrm{NLL}(2,4)+\mathrm{NLL}(3,4)+\mathrm{NLL}(4,4)+\mathrm{NLL}(5,4)}{5}}_{t=4} \\
= \frac{1}{5} \sum_{n=1}^5
\left[ \mathrm{NLL}(n,1)+\mathrm{NLL}(n,2)+\mathrm{NLL}(n,3)+\mathrm{NLL}(n,4) \right].
\end{aligned}
$$


### 3.6. Optimization via Mini-Batch SGD

We further show how to solve the pretrain problem to get the optimal parameter $\theta^*$. From the previous section, we know that our pretrain problems becomes

$$
\boxed{\;
\min_{\theta}\; H(P_{\text{data}}, P_{\theta})
\approx   \min_{\theta}\; \frac{1}{N}  \sum_{n=1}^N   \sum_{t = 1}^{L_{\max}} \left( - \log P_\theta( x^n_t \mid \vec{x}^n_{\lt t}) \right) .
\;}
$$

Let us define the global cross-entropy (NLL) loss function and per-data-sample sequence-level cross-entropy (NLL) loss function as

$$
\begin{aligned}
\mathcal{L}(\theta) := \frac{1}{N}  \sum_{n=1}^N   \sum_{t = 1}^{L_{\max}} \left( - \log P_\theta( x^n_t \mid \vec{x}^n_{\lt t}) \right), \\
\mathcal{L}^n(\theta) := \sum_{t = 1}^{L_{\max}} \left( - \log P_\theta( x^n_t \mid \vec{x}^n_{\lt t}) \right).
\end{aligned}
$$



Clearly, the global NLL loss is the average per-data-sample sequence-level NLL loss over all data samples, i.e.,

$$
\mathcal{L}(\theta) = \frac{1}{N}  \sum_{n=1}^N  \mathcal{L}^n(\theta).
$$

We generally uses gradient-descent algorithm to solve this optimization problem (even though the loss function may not be convex). Clearly, we have the following
per-data-sample gradient

$$
\nabla_\theta \mathcal{L}^n(\theta) = - \sum_{t = 1}^{L_{\max}} \frac{\nabla_\theta P_\theta(x^n_t \mid \vec{x}^n_{\lt t})}{P_\theta(x^n_t \mid \vec{x}^n_{\lt t})},
$$

and the global gradient

$$
\nabla_\theta \mathcal{L}(\theta) = \frac{1}{N}  \sum_{n=1}^N \nabla_\theta \mathcal{L}^n(\theta).
$$

Then we will generally update the parameter $\theta$ via the following gradient-descent algorithm,

$$
\theta_{t+1} = \theta_{t} - \eta_{t} \nabla_\theta \mathcal{L}(\theta_t) = \theta_{t} - \eta_{t} \cdot  \frac{1}{N}  \sum_{n=1}^N \nabla_\theta \mathcal{L}^n(\theta_t),
$$

where the learning rate {::nomarkdown}$\eta_{t} ${:/nomarkdown} should be small enough to guarantee convergence (if $\mathcal{L}(\theta)$ is convex over $\theta$).

However, as we mentioned before, $N$ is very large such that computing global gradient over all data samples is infeasible.
If we really do it, we will need to load all data samples into GPU for one single parameter update iteration/step. In practice,
we thus use stochastic-gradient-descent (SGD) algorithm to solve the optimization problem. Concretely,
We **randomly** divide all $N$ data samples into $M=\frac{N}{B}$ mini-batches of size $B$. Thus, each mini-batch has $B$ data samples.
We can then index the pretraining dataset by tuple (mini-batch-index, sample-index-inside-mini-batch) as
- Mini-Batch 1: data sample $(1,1)$, data sample $(1,2)$, ..., data sample $(1, B)$
- Mini-Batch 2: data sample $(2,1)$, data sample $(2,2)$, ..., data sample $(2, B)$
- ...
- Mini-Batch $M=\frac{N}{B}$:  data sample $(M,1)$, data sample $(M,2)$, ..., data sample $(M, B)$.

We still have in total $N$ data samples but organize them into 2D grid manner instead of 1D linear manner. Data sample $(b,i)$ of length {::nomarkdown}$L_{\max}${:/nomarkdown} is denoted as

$$
\vec{x}^{(b,i)} = \left(x^{(b,i)}_1, x^{(b,i)}_2, \cdots, x^{(b,i)}_{L_{\max}} \right), \qquad b=1,2,\cdots, M, i=1,2,\cdots, B
$$.

Now we have the following 2D loss function notations and gradient notations

$$
\begin{aligned}
\mathcal{L}^{(b,i)}(\theta) := \sum_{t = 1}^{L_{\max}} \left( - \log P_\theta( x^{(b,i)}_t \mid \vec{x}^{(b,i)}_{\lt t}) \right), \\
 \nabla_\theta \mathcal{L}^{(b,i)} (\theta) = - \sum_{t = 1}^{L_{\max}} \frac{\nabla_\theta P_\theta(x^{(b,i)}_t \mid \vec{x}^{(b,i)}_{\lt t})}{P_\theta(x^{(b,i)}_t \mid \vec{x}^{(b,i)}_{\lt t})}.
\end{aligned}
$$

Let us further define the per-mini-batch average loss and average gradient as

$$
\begin{aligned}
\mathcal{L}^{(b)}(\theta) =  \frac{1} {B} \sum_{i=1}^B  \mathcal{L}^{(b,i)}(\theta), \\
\nabla_\theta \mathcal{L}^{(b)} (\theta) = \frac{1}{B} \sum_{i=1}^B  \nabla_\theta \mathcal{L}^{(b,i)}(\theta).
\end{aligned}
$$

Then we can get the global loss function and gradient as follows

$$
\begin{aligned}
\mathcal{L}(\theta) &= \frac{1}{MB}  \sum_{b=1}^M \sum_{i=1}^B  \mathcal{L}^{(b,i)}(\theta) = \frac{1}{N}  \sum_{b=1}^M \sum_{i=1}^B  \mathcal{L}^{(b,i)}(\theta) \\
&= \frac{1}{M}  \sum_{b=1}^M  \left[ \frac{1} {B} \sum_{i=1}^B  \mathcal{L}^{(b,i)}(\theta) \right] = \frac{1}{M}  \sum_{b=1}^M \mathcal{L}^{(b)}(\theta), \\
\nabla_{\theta} \mathcal{L}(\theta) &= \frac{1}{MB}  \sum_{b=1}^M \sum_{i=1}^B  \nabla_\theta \mathcal{L}^{(b,i)}(\theta) = \frac{1}{N}  \sum_{b=1}^M \sum_{i=1}^B  \nabla_\theta  \mathcal{L}^{(b,i)}(\theta) \\
&= \frac{1}{M}  \sum_{b=1}^M \left[ \frac{1}{B} \sum_{i=1}^B  \nabla_\theta \mathcal{L}^{(b,i)}(\theta) \right] = \frac{1}{M}  \sum_{b=1}^M \nabla_\theta \mathcal{L}^{(b)} (\theta) .
\end{aligned}
$$

Clearly, the global gradient is the average of per-mini-batch gradient over all mini-batches. If we assume that all mini-batches follow the same distribution (not necessarily independent),
we have that

$$
\begin{aligned}
\mathbb{E} \left[ \mathcal{L}^{(b)}(\theta) \right] = \mathbb{E} \left[ \mathcal{L}^{(\tilde{b})}(\theta) \right] \\
=  \mathbb{E} \left[ \frac{1}{M}  \sum_{b=1}^M \mathcal{L}^{(b)}(\theta) \right] \\
=   \mathbb{E} \left[ \mathcal{L}(\theta) \right], \\
\mathbb{E} \left[ \nabla_\theta \mathcal{L}^{(b)} (\theta) \right] = \mathbb{E} \left[ \nabla_\theta \mathcal{L}^{(\tilde{b})}  (\theta) \right] \\
= \mathbb{E} \left[  \frac{1}{M}  \sum_{b=1}^M \nabla_\theta \mathcal{L}^{(b)} (\theta) \right] \\
= \mathbb{E} \left[ \nabla_{\theta} \mathcal{L}(\theta) \right].
\end{aligned}
$$

In other words, the per-mini-batch gradient (resp. loss) is an **unbiased estimator** for the global gradient (resp. loss). In addition, if we assume
that all data samples follow the same distribution and all independent, namely, i.i.d., and assume that any per-data-sample gradient has variance $\sigma^2$, then we have


$$
\begin{aligned}
\mathrm{Var} \left[ \nabla_\theta \mathcal{L}^{(b)} (\theta) \right]
&= \mathrm{Var} \left[ \frac{1}{B} \sum_{i=1}^B \nabla_\theta \mathcal{L}^{(b,i)} (\theta) \right] = \frac{\sigma^2}{B}, \\
\mathrm{Var} \left[ \nabla_\theta \mathcal{L} (\theta) \right]
&= \mathrm{Var} \left[ \frac{1}{MB} \sum_{b=1}^M \sum_{i=1}^B \nabla_\theta \mathcal{L}^{(b,i)} (\theta) \right] = \frac{\sigma^2}{MB}.
\end{aligned}
$$

If we let $B$ large enough, we can let {::nomarkdown}$\lim_{B \to \infty} \frac{\sigma^2}{B} = 0${:/nomarkdown}, and thus ensure
that both variance are close to zero. That is to say, in practice, the mini-batch size cannot be too small. However, too large $B$ results in OOM during pretraining.
We thus need to choose the largest one that avoids OOM. Now we know that the per-mini-batch gradient, which can be computed locally simply based on the mini-batch itself, provides a good estimation for the global gradient. Then the mini-batch SGD algorithm works as follows

$$
\theta_{t+1} = \theta_{t} - \eta_{t} \nabla_\theta \mathcal{L}^{(t)}(\theta_t) = \theta_{t} - \eta_{t} \cdot  \frac{1}{B}  \sum_{i=1}^B \nabla_\theta \mathcal{L}^{(t,i)}(\theta_t),
$$

> **Note**
>
> In practice (and also in this nanochat-ascend project), we do not use SGD but AdamW/Muon algorithms. In particular, we use AdamW to optimize the embeddings and scalars, and use Muon to optimize the matrix parameters. See the code in [`nanochat/optim.py`](https://github.com/leideng/nanochat-ascend/blob/main/nanochat/optim.py). However, their basic ideas are still SGD but has better performance. I will discuss them some days later when I have time. For the time being, it is enough to understand the mini-batch SGD algorithm.

### 3.7. Pretraining Running and Logs

To run pretraining locally, simply run

```bash
# Pretraining
bash runs/run_base_train.sh
```

It will execute the DDP distributed training process via `torchrun` as follows,

```bash
torchrun --nproc_per_node=16 --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --local-addr="$LOCAL_ADDR" -m scripts.base_train -- \
    --depth=20 \
    --aspect-ratio=64 \
    --head-dim=128 \
    --window-pattern=L \
    --max-seq-len=2048 \
    --device-batch-size=8 \
    --total-batch-size=-1 \
    --target-param-data-ratio=20 \
    --eval-every=1000 \
    --core-metric-every=2000 \
    --sample-every=2000 \
    --save-every=2000 \
    --run=$WANDB_RUN \
    --model-tag="d20"
```

After that, we can evaluate the pretrain model by running

```bash
# Evaluate base model
bash runs/run_base_eval.sh
```

The full d20 pretrain log is in [d20.pretrain.log](https://github.com/leideng/leideng.github.io/blob/main/_posts/d20.pretrain.log); a shortened version appears below.

<details markdown="1">
<summary>d20 Pretraining Log (head/middle/tail samples only)</summary>

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: Training will be less efficient without FA3
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Vocab size: 32,768
Model config:
{
  "sequence_len": 2048,
  "vocab_size": 32768,
  "n_layer": 20,
  "n_head": 10,
  "n_kv_head": 10,
  "n_embd": 1280,
  "window_pattern": "L"
}
Checkpoint directory: .cache/checkpoint/base_checkpoints/d20
torch.compile disabled (enforce_eager=true)
Parameter counts:
wte                     : 41,943,040
value_embeds            : 419,430,400
lm_head                 : 41,943,040
transformer_matrices    : 393,219,200
scalars                 : 40
total                   : 896,535,720
Estimated FLOPs per token: 3.240119e+09
Auto-computed optimal batch size: 1,048,576 tokens
Scaling LRs by 1.4142 for batch size 1,048,576 (reference: 524,288)
Scaling weight decay from 0.200000 to 0.071563 for depth 20
Scaling the LR for the AdamW parameters ∝1/√(1280/768) = 0.774597
Calculated number of iterations from target data:param ratio: 8,300
Total number of training tokens: 8,703,180,800
Tokens : Scaling params ratio: 20.00
Total training FLOPs estimate: 2.819934e+19
device_batch_size: 8
max_seq_len: 2048
ddp_world_size: 16
tokens_per_fwdbwd: 16384
world_tokens_per_fwdbwd: 262,144
total_batch_size: 1048576
total_batch_size // world_tokens_per_fwdbwd: 4
Tokens / micro-batch / rank: 8 x 2048 = 16,384
Tokens / micro-batch: 262,144
Total batch size 1,048,576 => gradient accumulation steps: 4
Step 00000 | Validation bpb: 3.167333
/data/ldeng/code/nanochat-ascend/.venv/lib/python3.11/site-packages/torch/autograd/__init__.py:221: UserWarning: Cannot create tensor with interal format while allow_internel_format=False, tensor will be created with base format. (Triggered internally at build/CMakeFiles/torch_npu.dir/compiler_depend.ts:338.)
  torch.ones_like(out, memory_format=torch.preserve_format)
step 00000/08300 (0.00%) | loss: 10.398064 | lrm: 1.00 | dt: 6331.86ms | tok/sec: 165,603 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00001/08300 (0.01%) | loss: 10.057295 | lrm: 1.00 | dt: 3708.63ms | tok/sec: 282,739 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00002/08300 (0.02%) | loss: 9.418337 | lrm: 1.00 | dt: 3578.22ms | tok/sec: 293,043 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00003/08300 (0.04%) | loss: 8.820613 | lrm: 1.00 | dt: 3576.68ms | tok/sec: 293,170 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00004/08300 (0.05%) | loss: 8.368130 | lrm: 1.00 | dt: 3575.19ms | tok/sec: 293,292 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00005/08300 (0.06%) | loss: 8.026400 | lrm: 1.00 | dt: 3575.66ms | tok/sec: 293,254 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00006/08300 (0.07%) | loss: 7.761053 | lrm: 1.00 | dt: 3575.40ms | tok/sec: 293,274 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00007/08300 (0.08%) | loss: 7.510014 | lrm: 1.00 | dt: 3575.44ms | tok/sec: 293,271 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00008/08300 (0.10%) | loss: 7.282737 | lrm: 1.00 | dt: 3575.11ms | tok/sec: 293,298 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00009/08300 (0.11%) | loss: 7.090680 | lrm: 1.00 | dt: 3576.52ms | tok/sec: 293,183 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00010/08300 (0.12%) | loss: 6.957730 | lrm: 1.00 | dt: 3576.21ms | tok/sec: 293,208 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.00m
step 00011/08300 (0.13%) | loss: 6.836487 | lrm: 1.00 | dt: 3577.65ms | tok/sec: 293,090 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.06m | eta: 494.3m
step 00012/08300 (0.14%) | loss: 6.745444 | lrm: 1.00 | dt: 3575.48ms | tok/sec: 293,268 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.12m | eta: 494.0m
step 00013/08300 (0.16%) | loss: 6.641261 | lrm: 1.00 | dt: 3575.60ms | tok/sec: 293,258 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.18m | eta: 493.9m
step 00014/08300 (0.17%) | loss: 6.560943 | lrm: 1.00 | dt: 3576.34ms | tok/sec: 293,198 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.24m | eta: 493.9m
step 00015/08300 (0.18%) | loss: 6.481138 | lrm: 1.00 | dt: 3576.74ms | tok/sec: 293,164 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.30m | eta: 493.8m
step 00016/08300 (0.19%) | loss: 6.401423 | lrm: 1.00 | dt: 3574.91ms | tok/sec: 293,314 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.36m | eta: 493.7m
step 00017/08300 (0.20%) | loss: 6.344683 | lrm: 1.00 | dt: 3577.03ms | tok/sec: 293,141 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.42m | eta: 493.7m
step 00018/08300 (0.22%) | loss: 6.301444 | lrm: 1.00 | dt: 3576.44ms | tok/sec: 293,189 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.48m | eta: 493.6m
step 00019/08300 (0.23%) | loss: 6.253164 | lrm: 1.00 | dt: 3576.17ms | tok/sec: 293,212 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.54m | eta: 493.6m
step 00020/08300 (0.24%) | loss: 6.214171 | lrm: 1.00 | dt: 3575.39ms | tok/sec: 293,275 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.60m | eta: 493.5m
step 00021/08300 (0.25%) | loss: 6.171882 | lrm: 1.00 | dt: 3576.25ms | tok/sec: 293,205 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.66m | eta: 493.5m
step 00022/08300 (0.27%) | loss: 6.139702 | lrm: 1.00 | dt: 3576.31ms | tok/sec: 293,200 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.72m | eta: 493.4m
step 00023/08300 (0.28%) | loss: 6.103848 | lrm: 1.00 | dt: 3575.25ms | tok/sec: 293,287 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.77m | eta: 493.3m
step 00024/08300 (0.29%) | loss: 6.085820 | lrm: 1.00 | dt: 3575.80ms | tok/sec: 293,242 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.83m | eta: 493.3m
step 00025/08300 (0.30%) | loss: 6.047692 | lrm: 1.00 | dt: 3577.51ms | tok/sec: 293,102 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.89m | eta: 493.2m
step 00026/08300 (0.31%) | loss: 6.008763 | lrm: 1.00 | dt: 3577.21ms | tok/sec: 293,126 | bf16_mfu: 0.00 | epoch: 1 | total time: 0.95m | eta: 493.2m
step 00027/08300 (0.33%) | loss: 5.975100 | lrm: 1.00 | dt: 3577.18ms | tok/sec: 293,129 | bf16_mfu: 0.00 | epoch: 1 | total time: 1.01m | eta: 493.1m
step 00028/08300 (0.34%) | loss: 5.951026 | lrm: 1.00 | dt: 3575.33ms | tok/sec: 293,280 | bf16_mfu: 0.00 | epoch: 1 | total time: 1.07m | eta: 493.0m
step 00029/08300 (0.35%) | loss: 5.936953 | lrm: 1.00 | dt: 3574.21ms | tok/sec: 293,373 | bf16_mfu: 0.00 | epoch: 1 | total time: 1.13m | eta: 493.0m
step 00030/08300 (0.36%) | loss: 5.913829 | lrm: 1.00 | dt: 3575.97ms | tok/sec: 293,228 | bf16_mfu: 0.00 | epoch: 1 | total time: 1.19m | eta: 492.9m
step 00031/08300 (0.37%) | loss: 5.881963 | lrm: 1.00 | dt: 3577.82ms | tok/sec: 293,076 | bf16_mfu: 0.00 | epoch: 1 | total time: 1.25m | eta: 492.9m
step 00032/08300 (0.39%) | loss: 5.864341 | lrm: 1.00 | dt: 3577.24ms | tok/sec: 293,124 | bf16_mfu: 0.00 | epoch: 1 | total time: 1.31m | eta: 492.8m
step 00033/08300 (0.40%) | loss: 5.840876 | lrm: 1.00 | dt: 3575.96ms | tok/sec: 293,229 | bf16_mfu: 0.00 | epoch: 1 | total time: 1.37m | eta: 492.7m
step 00034/08300 (0.41%) | loss: 5.816466 | lrm: 1.00 | dt: 3575.87ms | tok/sec: 293,236 | bf16_mfu: 0.00 | epoch: 1 | total time: 1.43m | eta: 492.7m
...
step 01998/08300 (24.07%) | loss: 2.964165 | lrm: 1.00 | dt: 3576.41ms | tok/sec: 293,192 | bf16_mfu: 0.00 | epoch: 1 | total time: 118.49m | eta: 375.6m
step 01999/08300 (24.08%) | loss: 2.957159 | lrm: 1.00 | dt: 3576.43ms | tok/sec: 293,190 | bf16_mfu: 0.00 | epoch: 1 | total time: 118.54m | eta: 375.5m
Step 02000 | Validation bpb: 0.889505
Evaluating: hellaswag_zeroshot (0-shot, type: multiple_choice)... accuracy: 0.3740 | centered: 0.1653 | time: 0.93s
Evaluating: jeopardy (10-shot, type: language_modeling)... accuracy: 0.0080 | centered: 0.0080 | time: 0.87s
Evaluating: bigbench_qa_wikidata (10-shot, type: language_modeling)... accuracy: 0.3900 | centered: 0.3900 | time: 1.05s
Evaluating: arc_easy (10-shot, type: multiple_choice)... accuracy: 0.5360 | centered: 0.3813 | time: 1.98s
Evaluating: arc_challenge (10-shot, type: multiple_choice)... accuracy: 0.2800 | centered: 0.0400 | time: 1.96s
Evaluating: copa (0-shot, type: multiple_choice)... accuracy: 0.6200 | centered: 0.2400 | time: 0.39s
Evaluating: commonsense_qa (10-shot, type: multiple_choice)... accuracy: 0.3680 | centered: 0.2100 | time: 2.00s
Evaluating: piqa (10-shot, type: multiple_choice)... accuracy: 0.6360 | centered: 0.2720 | time: 1.91s
Evaluating: openbook_qa (0-shot, type: multiple_choice)... accuracy: 0.3140 | centered: 0.0853 | time: 1.83s
Evaluating: lambada_openai (0-shot, type: language_modeling)... accuracy: 0.3160 | centered: 0.3160 | time: 1.78s
Evaluating: hellaswag (10-shot, type: multiple_choice)... accuracy: 0.3740 | centered: 0.1653 | time: 2.61s
Evaluating: winograd (0-shot, type: schema)... accuracy: 0.6044 | centered: 0.2088 | time: 1.01s
Evaluating: winogrande (0-shot, type: schema)... accuracy: 0.5100 | centered: 0.0200 | time: 1.91s
Evaluating: bigbench_dyck_languages (10-shot, type: language_modeling)... accuracy: 0.0860 | centered: 0.0860 | time: 1.92s
Evaluating: agi_eval_lsat_ar (3-shot, type: multiple_choice)... accuracy: 0.2261 | centered: 0.0326 | time: 1.33s
Evaluating: bigbench_cs_algorithms (10-shot, type: language_modeling)... accuracy: 0.3960 | centered: 0.3960 | time: 1.93s
Evaluating: bigbench_operators (10-shot, type: language_modeling)... accuracy: 0.1476 | centered: 0.1476 | time: 0.81s
Evaluating: bigbench_repeat_copy_logic (10-shot, type: language_modeling)... accuracy: 0.0312 | centered: 0.0312 | time: 0.12s
Evaluating: squad (10-shot, type: language_modeling)... accuracy: 0.0340 | centered: 0.0340 | time: 2.43s
Evaluating: coqa (0-shot, type: language_modeling)... accuracy: 0.1400 | centered: 0.1400 | time: 1.92s
Evaluating: boolq (10-shot, type: multiple_choice)... accuracy: 0.5380 | centered: -0.2158 | time: 2.70s
Evaluating: bigbench_language_identification (10-shot, type: multiple_choice)... accuracy: 0.2600 | centered: 0.1859 | time: 4.30s
Step 02000 | CORE metric: 0.1518
<|bos|>The capital of France is Paris, and the capital of France is Paris. Paris is the capital of France
<|bos|>The chemical symbol of gold is 24.5% gold. The symbol of silver is 24.5
<|bos|>If yesterday was Friday, then tomorrow will be today. The difference between the two is that tomorrow is a day, and today
<|bos|>The opposite of hot is cold. The opposite of cold is cold. The opposite of cold is cold.
<|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, and Saturn. The planets are
<|bos|>My favorite color is red. It’s a color that’s been around for centuries, and it’s
<|bos|>If 5*x + 3 = 13, then x is 5. If 5*x + 3 = 13, then
2026-03-30 01:22:12,807 - nanochat.checkpoint_manager - INFO - Saved model parameters to: .cache/checkpoint/base_checkpoints/d20/model_002000.pt
2026-03-30 01:22:12,808 - nanochat.checkpoint_manager - INFO - Saved metadata to: .cache/checkpoint/base_checkpoints/d20/meta_002000.json
2026-03-30 01:22:13,105 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: .cache/checkpoint/base_checkpoints/d20/optim_002000_rank0.pt
step 02000/08300 (24.10%) | loss: 2.974751 | lrm: 1.00 | dt: 3571.23ms | tok/sec: 293,618 | bf16_mfu: 0.00 | epoch: 1 | total time: 118.60m | eta: 375.5m
step 02001/08300 (24.11%) | loss: 2.975785 | lrm: 1.00 | dt: 3576.93ms | tok/sec: 293,149 | bf16_mfu: 0.00 | epoch: 1 | total time: 118.66m | eta: 375.4m
step 02002/08300 (24.12%) | loss: 2.973080 | lrm: 1.00 | dt: 3575.06ms | tok/sec: 293,303 | bf16_mfu: 0.00 | epoch: 1 | total time: 118.72m | eta: 375.4m
step 02003/08300 (24.13%) | loss: 2.978474 | lrm: 1.00 | dt: 3575.88ms | tok/sec: 293,235 | bf16_mfu: 0.00 | epoch: 1 | total time: 118.78m | eta: 375.3m
step 02004/08300 (24.14%) | loss: 2.961694 | lrm: 1.00 | dt: 3574.93ms | tok/sec: 293,313 | bf16_mfu: 0.00 | epoch: 1 | total time: 118.84m | eta: 375.2m
...
step 08286/08300 (99.83%) | loss: 2.553658 | lrm: 0.00 | dt: 3576.39ms | tok/sec: 293,194 | bf16_mfu: 0.00 | epoch: 1 | total time: 493.28m | eta: 0.8m
step 08287/08300 (99.84%) | loss: 2.543761 | lrm: 0.00 | dt: 3577.82ms | tok/sec: 293,076 | bf16_mfu: 0.00 | epoch: 1 | total time: 493.34m | eta: 0.8m
step 08288/08300 (99.86%) | loss: 2.552146 | lrm: 0.00 | dt: 3576.22ms | tok/sec: 293,207 | bf16_mfu: 0.00 | epoch: 1 | total time: 493.40m | eta: 0.7m
step 08289/08300 (99.87%) | loss: 2.547587 | lrm: 0.00 | dt: 3577.37ms | tok/sec: 293,113 | bf16_mfu: 0.00 | epoch: 1 | total time: 493.46m | eta: 0.7m
step 08290/08300 (99.88%) | loss: 2.556692 | lrm: 0.00 | dt: 3575.31ms | tok/sec: 293,282 | bf16_mfu: 0.00 | epoch: 1 | total time: 493.52m | eta: 0.6m
step 08291/08300 (99.89%) | loss: 2.574651 | lrm: 0.00 | dt: 3577.18ms | tok/sec: 293,129 | bf16_mfu: 0.00 | epoch: 1 | total time: 493.58m | eta: 0.5m
step 08292/08300 (99.90%) | loss: 2.592360 | lrm: 0.00 | dt: 3576.31ms | tok/sec: 293,200 | bf16_mfu: 0.00 | epoch: 1 | total time: 493.64m | eta: 0.5m
step 08293/08300 (99.92%) | loss: 2.588252 | lrm: 0.00 | dt: 3576.31ms | tok/sec: 293,200 | bf16_mfu: 0.00 | epoch: 1 | total time: 493.70m | eta: 0.4m
step 08294/08300 (99.93%) | loss: 2.564911 | lrm: 0.00 | dt: 3576.07ms | tok/sec: 293,220 | bf16_mfu: 0.00 | epoch: 1 | total time: 493.76m | eta: 0.4m
step 08295/08300 (99.94%) | loss: 2.562459 | lrm: 0.00 | dt: 3577.04ms | tok/sec: 293,141 | bf16_mfu: 0.00 | epoch: 1 | total time: 493.82m | eta: 0.3m
step 08296/08300 (99.95%) | loss: 2.578913 | lrm: 0.00 | dt: 3576.85ms | tok/sec: 293,156 | bf16_mfu: 0.00 | epoch: 1 | total time: 493.88m | eta: 0.2m
step 08297/08300 (99.96%) | loss: 2.559588 | lrm: 0.00 | dt: 3577.84ms | tok/sec: 293,075 | bf16_mfu: 0.00 | epoch: 1 | total time: 493.94m | eta: 0.2m
step 08298/08300 (99.98%) | loss: 2.558868 | lrm: 0.00 | dt: 3576.50ms | tok/sec: 293,184 | bf16_mfu: 0.00 | epoch: 1 | total time: 494.00m | eta: 0.1m
step 08299/08300 (99.99%) | loss: 2.556670 | lrm: 0.00 | dt: 3575.20ms | tok/sec: 293,291 | bf16_mfu: 0.00 | epoch: 1 | total time: 494.05m | eta: 0.1m
Step 08300 | Validation bpb: 0.781135
Evaluating: hellaswag_zeroshot (0-shot, type: multiple_choice)... accuracy: 0.4720 | centered: 0.2960 | time: 1.06s
Evaluating: jeopardy (10-shot, type: language_modeling)... accuracy: 0.1140 | centered: 0.1140 | time: 0.98s
Evaluating: bigbench_qa_wikidata (10-shot, type: language_modeling)... accuracy: 0.5360 | centered: 0.5360 | time: 1.01s
Evaluating: arc_easy (10-shot, type: multiple_choice)... accuracy: 0.6460 | centered: 0.5280 | time: 1.95s
Evaluating: arc_challenge (10-shot, type: multiple_choice)... accuracy: 0.3560 | centered: 0.1413 | time: 1.95s
Evaluating: copa (0-shot, type: multiple_choice)... accuracy: 0.6300 | centered: 0.2600 | time: 0.39s
Evaluating: commonsense_qa (10-shot, type: multiple_choice)... accuracy: 0.1840 | centered: -0.0200 | time: 1.98s
Evaluating: piqa (10-shot, type: multiple_choice)... accuracy: 0.7120 | centered: 0.4240 | time: 1.91s
Evaluating: openbook_qa (0-shot, type: multiple_choice)... accuracy: 0.3760 | centered: 0.1680 | time: 1.82s
Evaluating: lambada_openai (0-shot, type: language_modeling)... accuracy: 0.4000 | centered: 0.4000 | time: 1.81s
Evaluating: hellaswag (10-shot, type: multiple_choice)... accuracy: 0.4700 | centered: 0.2933 | time: 2.60s
Evaluating: winograd (0-shot, type: schema)... accuracy: 0.6447 | centered: 0.2894 | time: 1.03s
Evaluating: winogrande (0-shot, type: schema)... accuracy: 0.5100 | centered: 0.0200 | time: 1.93s
Evaluating: bigbench_dyck_languages (10-shot, type: language_modeling)... accuracy: 0.0880 | centered: 0.0880 | time: 1.92s
Evaluating: agi_eval_lsat_ar (3-shot, type: multiple_choice)... accuracy: 0.2957 | centered: 0.1196 | time: 1.17s
Evaluating: bigbench_cs_algorithms (10-shot, type: language_modeling)... accuracy: 0.4400 | centered: 0.4400 | time: 1.93s
Evaluating: bigbench_operators (10-shot, type: language_modeling)... accuracy: 0.1857 | centered: 0.1857 | time: 0.79s
Evaluating: bigbench_repeat_copy_logic (10-shot, type: language_modeling)... accuracy: 0.0000 | centered: 0.0000 | time: 0.14s
Evaluating: squad (10-shot, type: language_modeling)... accuracy: 0.3260 | centered: 0.3260 | time: 2.85s
Evaluating: coqa (0-shot, type: language_modeling)... accuracy: 0.2260 | centered: 0.2260 | time: 2.51s
Evaluating: boolq (10-shot, type: multiple_choice)... accuracy: 0.5720 | centered: -0.1263 | time: 2.86s
Evaluating: bigbench_language_identification (10-shot, type: multiple_choice)... accuracy: 0.2300 | centered: 0.1529 | time: 4.30s
Step 08300 | CORE metric: 0.2210
<|bos|>The capital of France is Paris. It is the largest city in France and the capital of the French department
<|bos|>The chemical symbol of gold is Au. It is a soft, malleable, ductile, and sil
<|bos|>If yesterday was Friday, then tomorrow will be Saturday. If yesterday was Sunday, then tomorrow will be Monday. If yesterday was
<|bos|>The opposite of hot is cold. The opposite of cold is hot. The opposite of hot is cold.
<|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune,
<|bos|>My favorite color is green. I love the color green. I love the color green. I love
<|bos|>If 5*x + 3 = 13, then x is 5. If 5*x + 3 = 13, then
2026-03-30 07:44:00,759 - nanochat.checkpoint_manager - INFO - Saved model parameters to: .cache/checkpoint/base_checkpoints/d20/model_008300.pt
2026-03-30 07:44:00,760 - nanochat.checkpoint_manager - INFO - Saved metadata to: .cache/checkpoint/base_checkpoints/d20/meta_008300.json
2026-03-30 07:44:01,082 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: .cache/checkpoint/base_checkpoints/d20/optim_008300_rank0.pt
Peak memory usage: 33720.95MiB
Total training time: 494.05m
Minimum validation bpb: 0.781135
```

</details>

Base-model performance is documented in the [nanochat-ascend README](https://github.com/leideng/nanochat-ascend/blob/main/README.md). I also reproduce the key numbers here.

| Reference | Source |
| --- | --- |
| nanochat-ascend d20 — pretraining | [base-model-training.md](https://github.com/leideng/nanochat-ascend/blob/v0.1/dev/d20_eval_results/base-model-training.md) |
| nanochat-ascend d20 — base evaluation | [base-model-evaluation.md](https://github.com/leideng/nanochat-ascend/blob/v0.1/dev/d20_eval_results/base-model-evaluation.md) |
| nanochat-ascend d32 — pretraining | [base-model-training (iter 16k–17k).md](https://github.com/leideng/nanochat-ascend/blob/v0.1/dev/d32_eval_results/base-model-training-iter-from-16000-to-17000.md) |
| nanochat-ascend d32 — base evaluation | [base-model-evaluation.md](https://github.com/leideng/nanochat-ascend/blob/v0.1/dev/d32_eval_results/base-model-evaluation.md) |
| Karpathy d20 (upstream speedrun) | [nanochat GitHub discussion #1](https://github.com/karpathy/nanochat/discussions/1) |
| Karpathy d32 (upstream \$1000 run) | [nanochat GitHub discussion #8](https://github.com/karpathy/nanochat/discussions/8) |



The table below compares base pretraining runs against upstream nanochat. Depth labels do not denote the same architecture: nanochat-ascend uses a wider configuration at a given depth, so parameter counts and compute differ from Karpathy’s runs. Our vocabulary size is 32,768 ($2^{15}$) versus 65,536 ($2^{16}$) upstream, and we train with a smaller tokens:params ratio than 20 to speed up training. The results show that BPB and core-task evaluation are comparable to Karpathy's runs.

| Metric | nanochat-ascend d20 | Karpathy d20 | nanochat-ascend d32 | Karpathy d32 |
| --- | --- | --- | --- | --- |
| **Parameters** | 896,535,720 | 560,988,160 | 2,818,580,544 | 1,879,048,192 |
| **Vocab size** | 32,768 (2^15) | 65,536 (2^16) | 32,768 (2^15) | 65,536 (2^16) |
| **Training Model** | Eager | `torch.compile` (CUDA) | Eager | `torch.compile` (CUDA) |
| **Training tokens** | 8,703,180,800 | 11,219,763,200 | 35,651,584,000 | 37,580,963,840 |
| **Tokens∶params** | 9.7 | 20.0 | 12.6 | 20.0 |
| **Iterations** | 8,300 | 21,400 | 17,000 | 71,680 |
| **Total training FLOPs** | 2.82×10¹⁹ | 3.92×10¹⁹ | 4.16×10²⁰ | 4.54×10²⁰ |
| **Final val BPB** | 0.7811 | 0.81 | 0.7026 | 0.7236 |
| **CORE** (`base_eval`) | 0.2167 | 0.2219 | 0.2881 | 0.3168 |


## References

- [1] Philip Whittington, Gregor Bachmann, and Tiago Pimentel. *Tokenisation is NP-Complete.* ACL, 2025. [PDF](https://aclanthology.org/2025.acl-long.1365.pdf)
- [2] V. Zouhar, C. Meister, J. Gastaldi, L. Du, T. Vieira, M. Sachan, and R. Cotterell. *A Formal Perspective on Byte-Pair Encoding.* ACL Findings, 2023. [PDF](https://aclanthology.org/2023.findings-acl.38v2.pdf)
- [3] Philip Gage. *A New Algorithm for Data Compression.* The C Users Journal, 12(2):23–38, 1994. [Article](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM)
- [4] Rico Sennrich, Barry Haddow, and Alexandra Birch. *Neural Machine Translation of Rare Words with Subword Units.* ACL, 2016. [Paper](https://aclanthology.org/P16-1162/)
- [5] Li Du, Lucas Torroba Hennigen, Tiago Pimentel, Clara Meister, Jason Eisner, and Ryan Cotterell. *A Measure-Theoretic Characterization of Tight Language Models.* ACL, 2023. [PDF](https://aclanthology.org/2023.acl-long.543.pdf)
- [6] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. *Language Models are Unsupervised Multitask Learners.* OpenAI, 2019. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [7] Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. *Sequence to Sequence Learning with Neural Networks.* NeurIPS, 2014. [PDF](https://arxiv.org/pdf/1409.3215)
- [8] Gian Wiher, Clara Meister, and Ryan Cotterell. *On Decoding Strategies for Neural Text Generators.* TACL, 2022. [Paper](https://aclanthology.org/2022.tacl-1.58/)
- [9] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. *Scaling Laws for Neural Language Models.* arXiv, 2020. [Paper](https://arxiv.org/abs/2001.08361)
- [10] DeepSeek-AI. *DeepSeek-V4.* 2025. [PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)