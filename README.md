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

Local copies of the Kaggle runs, with outputs, are kept in `notebooks/kaggle_runs/` (gitignored).

## Setup as run

Full mode (`DEBUG=False`), Kaggle GPU T4 x2.

- Corpus: 299,985 passages (positives + one stored negative per core query + 100k random distractors).
- Queries: val 5,000, test 5,000, alignment 200k (train + other, never val/test), refinement 90k (train).
- Stored negative = `negative_passages[0]` of the dataset (not verified to be BM25).

| stage | data | epochs | batch per GPU (global) | lr / head_lr | trainable params |
|---|---|---|---|---|---|
| align, 28 layers (LoRA r=16 + head 1024→256) | 200k queries | 1 | 128 (256) | 2e-4 / 1e-3 | 10.35M |
| prune 28→14 + re-align (full fine-tune) | 200k queries | 1 | 64 (128) | 2e-5 / 1e-4 | 375.81M |
| prune 14→7 + re-align (full fine-tune) | 200k queries | 1 | 64 (128) | 2e-5 / 1e-4 | 265.69M |
| contrastive refinement, n_neg=3, margin=0.2 | 90k queries | 1 | 96 (192) | 5e-5 / 1e-4 | as above per student |

- Kept layers: `prog_l14` = `[0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 21, 23, 25, 27]`, `prog_l7` = `[0, 4, 8, 12, 19, 23, 27]`.
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

## Open items to check

- 03's own evaluation of the `_cr` models on test differs from 04's re-evaluation (aligned_cr nDCG@10 0.7306 in 03
  vs 0.7276 in 04; prog_l7_cr 0.6015 vs 0.6077, R@100 0.9377 vs 0.9424). This README reports 04's numbers.
- 04's ablation markdown mentions "untrained student (own index) → aligned", but the cell doesn't compute it.
- The Kaggle copy of notebook 01 has no saved outputs; the teacher row comes from 02 / 04.

## Conclusions

