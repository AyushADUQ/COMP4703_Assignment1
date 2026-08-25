# COMP4703 Assignment 1: The Biography Fact Extractor

**Released:** Week 2 · **Due:** end of Week 5 · **Weight:** 15% of final grade

- **`COMP4703_Assignment1_Stub.ipynb`** (same directory): a single
  self-contained Jupyter notebook. 
  Everything runs in one notebook, checks print ✅/❌ inline.

## Overview

Across Weeks 1-4 we've built up, on paper, the classic statistical NLP
toolkit: n-gram language models, Hidden Markov Model POS tagging, and
CFG-based parsing. This assignment asks you to implement most of this toolkit
**from scratch in Python** and wire it into a single pipeline that solves
a version of the running example from lecture:

> "Albert Einstein was born in Ulm, in the Kingdom of Württemberg, on 14
> March 1879" → `{who: "Albert Einstein", did_what: "was born",
> where: ["Ulm"], when: "14 March 1879"}`

You will not be using spaCy, nltk's built-in taggers/parsers, or any
pretrained model to do the actual language modelling / tagging / parsing
— the whole point is to implement the chain rule, MLE, add-k smoothing,
Viterbi, and a CFG parser yourselves, so that you *understand* what a
library like spaCy is doing under the hood before you're allowed to
treat it as a black box later in the course.

## Learning objectives

| Part | Week | You will implement |
|---|---|---|
| A | 2 | n-gram counting, MLE, add-k smoothing, perplexity, Shannon sampling |
| B | 3 | HMM parameter estimation, the Viterbi algorithm |
| C | 3+4 | rule-based NER, end-to-end pipeline integration and evaluation |

## Setup

```bash
cd COMP4703_Assignment1 
conda create -n "COMP4703A1" python=3.10
conda activate COMP4703A1
pip install -r requirements.txt
python -m nltk.downloader brown treebank 
jupyter notebook 
```

## Repository structure

```
COMP4703_Assignment1/
├── data/
│   ├── sam_corpus.txt             Toy corpus from the Week 2 slides.
│   ├── toy_tagged_corpus.txt      Toy tagged corpus for Part B demo.
│   ├── biography_sentences.json   Gold evaluation set for Part D (20 examples).
│   └── challenge_sentences.json   Stretch evaluation set (varied sentence structure).
├── COMP4703_Assignment1.ipynb     Everything you need for the assignment.
└── requirements.txt               The conda/pip environment settings. 
└── README.md                      The instructions you are reading now. 
```

Every `raise NotImplementedError` in the jupyter notebook is something 
you must write. Function/class docstrings specify the **exact contract** 
the tests check
— Read them carefully before you start; they tell you argument shapes,
return types, and edge-case handling.

**Calibrate your expectations before you start debugging.** A correct
reference implementation, trained on the full Penn Treebank, scores
roughly: `did_what_accuracy` and `when_accuracy` near 1.0, but
`who_accuracy` only around **0.1–0.25**, and `where_recall` around
**0.35**. This is not a bug to chase away — it's a genuine, well-known
weakness of bigram HMMs: Treebank is Wall Street Journal financial news,
and "Albert Einstein" is an out-of-vocabulary bigram of out-of-vocabulary
words relative to that domain. With no reliable emission signal, Viterbi
falls back almost entirely on transition probabilities, and "PROPN
PROPN" is simply not as dominant a transition in WSJ as, say, "PRON AUX"
("it was...", "he said..."), so `find_person`/`find_locations` frequently
grab the wrong span or nothing at all. **This domain-mismatch finding —
not a high accuracy number — is the intended result of Part D**, and
your report's error analysis should center on it. One concrete, gradeable
way to improve on it: mix a small hand-tagged set of biography-style
sentences (`data/toy_tagged_corpus.txt` is a starting point — extend it)
into your training data, oversampling if needed, and report how much
that in-domain "seed data" moves the needle. This is a legitimate,
widely-used technique (domain adaptation via a small labelled seed set),
not a hack.

## What to submit
1. Your completed jupyter notebook along with the snaphot state.
2. So, when you start working, PLEASE rename folder `COMP4703_Assignment1/' 
   as your Student ID. So, something like sXXXXXX/.
3. Now create a zip file CONTAINING this folder. When we unzip it for 
   marking, it should create a folder + file like: 
   `sXXXXXX/COMP4703_Assignment1.ipynb'.

Why? Because this will be a unique identifer and not overwrite other student solutions during marking, and this will ideally capture the hidden folder capturing the state of your checkpointed runs called `.ipynb_checkpoints'.

## Academic integrity
You may use library functions for tokenisation, corpus loading, basic
data structures, and numerical utilities (`math.log2`, `random.choices`,
etc.). You may **not** use any function that estimates n-gram
probabilities, decodes an HMM, tags POS, chunks, or parses on your
behalf (`nltk.lm`, `nltk.tag`, `nltk.chunk`, `nltk.parse`, `spacy`,
`sklearn-crfsuite`, or equivalents). If you're unsure whether a library
call crosses this line, ask on the course forum before using it.

## Grading rubric

| Component | Weight | Focus |
|---|---|---|
| Part A: N-gram LM + smoothing + perplexity | 5 marks | correctness of chain rule/MLE, smoothing, log-space stability, real-corpus results |
| Part B: HMM POS tagger + Viterbi | 5 marks | correct DP implementation, accuracy vs. baseline on real data |
| Part C: NER + Integration | 5 marks | pipeline wiring, honest per-slot precision/recall, error analysis |

## Tips
- Work in log-space everywhere probabilities get multiplied across a
  sequence (LM sentence probabilities, Viterbi) — this is not optional
  once sentences get long, it's the difference between working code and
  silent underflow to 0.0.
- Get the toy-corpus tests passing before touching a real corpus. The
  toy fixtures are small enough to verify by hand against the lecture
  slides; a real corpus is not.
- Part C depends on Parts B (and A); if  the `test_pipeline` is
  failing, check whether your POS tagger  passes all of the tests first.
