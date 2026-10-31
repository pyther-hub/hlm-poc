# hlm-poc: experiment notes

A small student query encoder (Qwen3-Embedding-0.6B) retrieves against a frozen teacher document index
(Qwen3-Embedding-4B, d=256). Code docs are in [doc.md](doc.md).

## Notebooks

| step | Kaggle run (with outputs) | repo notebook | what it does |
|---|---|---|---|
| 01 | [01-generate-teacher-embeddings](https://www.kaggle.com/code/tensorpanda231/01-generate-teacher-embeddings) | [01_generate_teacher_embeddings.ipynb](notebooks/01_generate_teacher_embeddings.ipynb) | build MS MARCO data, embed corpus + queries with the frozen 4B teacher |
| 02 | [02-student-alignment-and-pruning](https://www.kaggle.com/code/tensorpanda231/02-student-alignment-and-pruning) | [02_student_alignment_and_pruning.ipynb](notebooks/02_student_alignment_and_pruning.ipynb) | align the 0.6B student to teacher query vectors, prune 28 → 14 → 7 layers with re-alignment |
| 03 | [03-contrastive-refinement](https://www.kaggle.com/code/tensorpanda231/03-contrastive-refinement) | [03_contrastive_refinement.ipynb](notebooks/03_contrastive_refinement.ipynb) | contrastive refinement of every student against the frozen teacher doc vectors |
| 04 | [04-retrieval-evaluation-and-ablation](https://www.kaggle.com/code/tensorpanda231/04-retrieval-evaluation-and-ablation) | [04_retrieval_evaluation_and_ablation.ipynb](notebooks/04_retrieval_evaluation_and_ablation.ipynb) | evaluate all variants on test, efficiency, ablations |
| 05 | [05-bge-small-align-and-refine](https://www.kaggle.com/code/tensorpanda231/05-bge-small-align-and-refine) | [05_bge_small_align_and_refine.ipynb](notebooks/05_bge_small_align_and_refine.ipynb) | BGE students (small / base / large): full fine-tune align + contrastive refinement |

Local copies of the Kaggle runs, with outputs, are kept in `notebooks/kaggle_runs/` (gitignored).

## Setup as run

- Corpus: 299,985 passages (positives + one stored negative per core query + 100k random distractors).
- Queries: val 5,000, test 5,000, alignment 200k (train + other, never val/test), refinement 90k (train).
- Stored negative = `negative_passages[0]` of the dataset (not verified to be BM25).

| stage | data | epochs | batch per GPU (global) | lr / head_lr | trainable params |
|---|---|---|---|---|---|
| align, 28 layers (LoRA r=16 + head 1024→256) | 200k queries | 1 | 128 (256) | 2e-4 / 1e-3 | 10.35M |
| prune 28→14 + re-align (full fine-tune) | 200k queries | 1 | 64 (128) | 2e-5 / 1e-4 | 375.81M |
| prune 14→7 + re-align (full fine-tune) | 200k queries | 1 | 64 (128) | 2e-5 / 1e-4 | 265.69M |
| contrastive refinement, n_neg=3, margin=0.2 | 90k queries | 1 | 96 (192) | 5e-5 / 1e-4 | as above per student |

- Refinement ran with `n_neg=3, margin=0.2`; the repo defaults are `n_neg=1, margin=0.1`.

## Results

All numbers are from notebook 04: student queries vs the frozen teacher corpus, top-100 retrieval.
retention = nDCG@10 / teacher nDCG@10. `_cr` = after contrastive refinement.

### Retrieval quality (test)

| run | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | MRR@10 | nDCG@10 | retention |
|---|---|---|---|---|---|---|---|---|---|
| teacher | 0.6396 | 0.8969 | 0.9457 | 0.9716 | 0.9865 | 0.9928 | 0.7639 | 0.8063 | 1.0000 |
| aligned | 0.5195 | 0.7927 | 0.8636 | 0.9099 | 0.9521 | 0.9707 | 0.6486 | 0.6975 | 0.8651 |
| prog_l14 | 0.4595 | 0.7414 | 0.8136 | 0.8673 | 0.9214 | 0.9479 | 0.5900 | 0.6410 | 0.7951 |
| prog_l7 | 0.3716 | 0.6319 | 0.7253 | 0.7953 | 0.8685 | 0.9098 | 0.4947 | 0.5470 | 0.6784 |
| aligned_cr | 0.5521 | 0.8210 | 0.8867 | 0.9283 | 0.9646 | 0.9775 | 0.6805 | 0.7276 | 0.9024 |
| prog_l14_cr | 0.4890 | 0.7684 | 0.8416 | 0.8985 | 0.9426 | 0.9623 | 0.6189 | 0.6698 | 0.8307 |
| prog_l7_cr | 0.4246 | 0.7024 | 0.7921 | 0.8549 | 0.9121 | 0.9424 | 0.5531 | 0.6077 | 0.7537 |

<details>
<summary>Retrieval quality (val)</summary>

| run | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | MRR@10 | nDCG@10 | retention |
|---|---|---|---|---|---|---|---|---|---|
| teacher | 0.6349 | 0.8941 | 0.9405 | 0.9719 | 0.9897 | 0.9937 | 0.7591 | 0.8015 | 1.0000 |
| aligned | 0.5075 | 0.7899 | 0.8635 | 0.9116 | 0.9521 | 0.9701 | 0.6419 | 0.6926 | 0.8641 |
| prog_l14 | 0.4552 | 0.7316 | 0.8105 | 0.8721 | 0.9261 | 0.9501 | 0.5862 | 0.6370 | 0.7948 |
| prog_l7 | 0.3574 | 0.6322 | 0.7153 | 0.7869 | 0.8642 | 0.9079 | 0.4845 | 0.5371 | 0.6702 |
| aligned_cr | 0.5374 | 0.8145 | 0.8820 | 0.9280 | 0.9635 | 0.9786 | 0.6701 | 0.7188 | 0.8968 |
| prog_l14_cr | 0.4782 | 0.7629 | 0.8392 | 0.8980 | 0.9368 | 0.9590 | 0.6136 | 0.6652 | 0.8299 |
| prog_l7_cr | 0.4204 | 0.6994 | 0.7865 | 0.8443 | 0.9098 | 0.9376 | 0.5514 | 0.6050 | 0.7549 |

</details>

**Observations:**

### Ablation (test)

| step | from → to | nDCG@10 | R@100 |
|---|---|---|---|
| prune 28→14 | aligned → prog_l14 | 0.697 → 0.641 | 0.971 → 0.948 |
| prune 14→7 | prog_l14 → prog_l7 | 0.641 → 0.547 | 0.948 → 0.910 |
| refine | aligned → aligned_cr | 0.697 → 0.728 | 0.971 → 0.978 |
| refine | prog_l14 → prog_l14_cr | 0.641 → 0.670 | 0.948 → 0.962 |
| refine | prog_l7 → prog_l7_cr | 0.547 → 0.608 | 0.910 → 0.942 |

**Observations:**

### Efficiency (query encoding, fp16, one T4)

Documents are pre-embedded, so only query encoding is timed. Memory = parameter bytes in fp16.
The `_cr` models have the same size and speed as their base model (within noise).

| model | params | fp16 MB | ms/query bs=1 | bs=8 | bs=32 | bs=128 |
|---|---|---|---|---|---|---|
| teacher_4b | 4022M | 8044 | 66.752 | 15.951 | 14.251 | 14.200 |
| aligned | 606M | 1212 | 77.069 | 9.134 | 2.366 | 1.406 |
| prog_l14 | 376M | 752 | 26.577 | 3.116 | 0.806 | 0.583 |
| prog_l7 | 266M | 531 | 14.020 | 1.710 | 0.474 | 0.307 |
| aligned_cr | 606M | 1212 | 77.281 | 9.203 | 2.381 | 1.407 |
| prog_l14_cr | 376M | 752 | 26.550 | 3.088 | 0.807 | 0.580 |
| prog_l7_cr | 266M | 531 | 13.994 | 1.730 | 0.466 | 0.308 |

Speedup vs teacher_4b (ms/query ratio):

| model | bs=1 | bs=8 | bs=32 | bs=128 |
|---|---|---|---|---|
| aligned | 0.87x | 1.75x | 6.02x | 10.10x |
| prog_l14 | 2.51x | 5.12x | 17.68x | 24.34x |
| prog_l7 | 4.76x | 9.33x | 30.07x | 46.32x |
| aligned_cr | 0.86x | 1.73x | 5.99x | 10.09x |
| prog_l14_cr | 2.51x | 5.17x | 17.67x | 24.48x |
| prog_l7_cr | 4.77x | 9.22x | 30.56x | 46.15x |

**Observations:**

### Mean cos(q, d+) on val

| teacher | aligned | prog_l14 | prog_l7 | aligned_cr | prog_l14_cr | prog_l7_cr |
|---|---|---|---|---|---|---|
| 0.762 | 0.704 | 0.687 | 0.654 | 0.670 | 0.635 | 0.604 |

**Observations:**

## BGE students (notebook 05)

Kaggle run: [05-bge-small-align-and-refine](https://www.kaggle.com/code/tensorpanda231/05-bge-small-align-and-refine).
Local copies: `notebooks/kaggle_runs_final/bge-runs/05-bge-{small,base,large}-align-and-refine.ipynb`.
Same teacher index, data and eval as above. Student = `BAAI/bge-*-en-v1.5`, CLS pooling, linear head → 256,
**full fine-tune** (no LoRA, no pruning), `QUERY_PREFIX=""`. Two stages: align (L2), then contrastive refinement
(`tau=0.05, margin=0.1, n_neg=1, l2_weight=0.1`).

### Setup as run

| student | params | align: epochs / global batch / lr / head_lr | align steps | refine: epochs / global batch / lr / head_lr | refine steps | align min | refine min |
|---|---|---|---|---|---|---|---|
| bge-small | 33.31M | 20 / 2048 / 1e-4 / 3e-4 | 1940 | 20 / 2048 / 3e-5 / 3e-5 | 860 | 17.7 | 8.7 |
| bge-base | 109.09M | 12 / 1024 / 7e-5 / 3e-4 | 2340 | 12 / 1024 / 7e-5 / 3e-5 | 1044 | 24.7 | 11.7 |
| bge-large | 334.35M | 8 / 256 / 1e-4 / 3e-4 | 6248 | 8 / 256 / 3e-5 / 3e-5 | 1500 of 2808 (crashed) | 65.4 | 16.0 |

- bge-large refinement died at step 1500 (NCCL collective timeout while rank 0 saved the step-1500 checkpoint,
  rank 1 SIGABRT). No `bge_aligned_cr/final`, so no eval rows for it; its refine numbers below are the in-training
  val evals (same val set and corpus as the final eval: for small/base the logged val nDCG@10 matches the final
  table to 4 decimals).

### Retrieval quality (test)

| run | R@10 | R@100 | MRR@10 | nDCG@10 | retention |
|---|---|---|---|---|---|
| teacher | 0.9457 | 0.9928 | 0.7639 | 0.8063 | 1.0000 |
| bge-small aligned | 0.7572 | 0.9266 | 0.5166 | 0.5714 | 0.7087 |
| bge-small aligned_cr | 0.8171 | 0.9532 | 0.5783 | 0.6327 | 0.7848 |
| bge-base aligned | 0.8629 | 0.9709 | 0.6389 | 0.6905 | 0.8564 |
| bge-base aligned_cr | 0.8850 | 0.9760 | 0.6601 | 0.7121 | 0.8832 |
| bge-large aligned | n/a | n/a | n/a | n/a | n/a |
| bge-large aligned_cr | n/a | n/a | n/a | n/a | n/a |

<details>
<summary>Retrieval quality (val)</summary>

| run | R@10 | R@100 | MRR@10 | nDCG@10 | retention |
|---|---|---|---|---|---|
| teacher | 0.9405 | 0.9937 | 0.7591 | 0.8015 | 1.0000 |
| bge-small aligned | 0.7556 | 0.9325 | 0.5184 | 0.5722 | 0.7139 |
| bge-small aligned_cr | 0.8240 | 0.9541 | 0.5782 | 0.6347 | 0.7919 |
| bge-base aligned | 0.8510 | 0.9685 | 0.6245 | 0.6766 | 0.8442 |
| bge-base aligned_cr | 0.8719 | 0.9736 | 0.6442 | 0.6971 | 0.8697 |

</details>

**Observations:**

### Contrastive refinement gain vs student size (val, from training logs)

Val is the only split all three have. `aligned` = last align eval, `cr last` = last refine eval,
`cr best` = best refine eval (step). Qwen3-0.6B (LoRA, 10.35M trainable) rows from notebook 04 for reference.

| student | trainable | aligned nDCG@10 | cr last nDCG@10 | cr best nDCG@10 (step) | Δ last | Δ best | aligned R@10 | cr last R@10 |
|---|---|---|---|---|---|---|---|---|
| bge-small | 33M | 0.5720 | 0.6347 | 0.6347 (700) | +0.0627 | +0.0627 | 0.7556 | 0.8242 |
| bge-base | 109M | 0.6768 | 0.6971 | 0.7002 (200) | +0.0203 | +0.0234 | 0.8510 | 0.8717 |
| bge-large | 334M | 0.7467 | 0.7456 (step 1500) | 0.7502 (400) | −0.0011 | +0.0035 | 0.9045 | 0.9097 |
| qwen3-0.6B LoRA (04) | 10.35M | 0.6926 | 0.7188 | n/a | +0.0262 | n/a | 0.8635 | 0.8820 |

Refine val nDCG@10 at the first eval (step 100): small 0.6245, base 0.6993, large 0.7491.

**Observations:**

### L2 alignment alone (val)

retention = val nDCG@10 / teacher val nDCG@10 (0.8015). cos = mean cos(student q, teacher q) on val.

| student | aligned nDCG@10 | retention | cos after align | cos after refine (last) |
|---|---|---|---|---|
| bge-small | 0.5720 | 0.7137 | 0.8127 | 0.7566 |
| bge-base | 0.6768 | 0.8444 | 0.8609 | 0.7977 |
| bge-large | 0.7467 | 0.9316 | 0.9129 | 0.8715 |
| qwen3-0.6B LoRA (04) | 0.6926 | 0.8641 | n/a | n/a |

**Observations:**

### Efficiency (query encoding, fp16, one T4)

| model | params | ms/query bs=1 | bs=32 |
|---|---|---|---|
| bge-small aligned | 33M | 8.019 | 0.353 |
| bge-small aligned_cr | 33M | 8.099 | 0.357 |
| bge-base aligned | 109M | 8.681 | 0.390 |
| bge-base aligned_cr | 109M | 8.561 | 0.381 |
| bge-large | 334M | n/a | n/a |

**Observations:**

## Conclusions

