# blog.md: what to add before publishing

Review notes on the first draft of [blog.md](blog.md). Not part of the blog.

## 1. Make the three phases reproducible

For each phase, document:

- Exact loss function and optimization objective.
- Learning rate, batch size, epochs and optimizer.
- Number of training examples.
- Checkpoint selection and recovery strategy.

Whether L2 alignment is applied to queries only or to both queries and passages: queries only. The student is a
query encoder, and documents are embedded once by the frozen teacher. State this explicitly in the blog.

## 2. Add a consolidated results table

This is probably the most important addition.

| Model variant       | Parameters | Recall\@10 | nDCG\@10 | MRR\@10 | Latency |
| ------------------- | ---------- | ---------- | -------- | ------- | ------- |
| Teacher             | 4B         |            |          |         |         |
| Initial student     | 0.6B       |            |          |         |         |
| Aligned student     |            |            |          |         |         |
| Pruned student      |            |            |          |         |         |
| Contrastive refined |            |            |          |         |         |

Keep the same document index and evaluation queries across all variants.

## 3. Explain the experimental controls

Document the random seed, split construction, query overlap checks, negative sampling strategy, and how the
retrieval corpus was constructed. Also confirm that all test positives are present in the retrieval corpus.

## 4. Add an efficiency comparison

Report:

- Parameter reduction percentage.
- Model memory consumption.
- Single-query latency (batch size 1).
- Throughput at larger batch sizes.

Separate query encoding latency from document indexing and retrieval latency.

## 5. Add an ablation study

Compare alignment alone, alignment + pruning, alignment + pruning + recovery, and the complete recipe with
contrastive refinement. This will help establish which phase contributes to the final performance.

## Methodological qualifications

- The samples come from the Hugging Face dataset's `train` split, so describe the 5K validation and 5K test sets
  as internal held-out splits, not official MS MARCO benchmark test results. The dataset listing shows
  approximately 401K training records with query, positive-passage and negative-passage fields.
- The LEAF paper supports L2 representation alignment experimentally, but it does not prove that this particular
  pruning and contrastive-refinement recipe will work. Its own experiments found that simple L2 approximation was
  effective for its setup.

## MRR formula

MRR must be averaged over queries:

```
MRR@10 = (1 / |Q|) * Σ_{q ∈ Q} ( 1 / r_q   if r_q <= 10
                                 0         otherwise )
```

Here, `r_q` is the rank of the first relevant result for query `q`.

## Suggested priority for the next iteration

Do not expand the prose yet. First add these three things:

1. A methodology diagram: Teacher → L2 alignment → pruning and recovery → contrastive refinement.
2. A complete results comparison: every student architecture, pruning strategy and training phase.
3. An analysis section: explain why L2 alignment plateaus with smaller models and whether contrastive refinement
   recovers retrieval performance.

The most interesting potential research contribution is the interaction between representation-level alignment and
retrieval-level contrastive supervision under constrained student capacity. Make this the central theme of the
article rather than presenting the work simply as a model compression exercise.
