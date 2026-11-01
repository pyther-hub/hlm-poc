# hlm-poc

Can a small query encoder search a large model's document index? This proof of concept trains a small student
model to encode queries into the embedding space of a frozen teacher, Qwen3-Embedding-4B (d=256). The teacher embeds
the documents once, and the student encodes only the queries, so the student is cheap to run online.

The recipe has three phases:

1. **L2 alignment.** The student (Qwen3-Embedding-0.6B, or another small embedding model) is trained to reproduce
   the teacher's query vectors.
2. **Structured pruning and recovery.** The aligned student is cut from 28 to 14 to 7 Transformer layers, and each
   pruned model is re-aligned with the same L2 objective.
3. **Contrastive refinement.** Each student is fine-tuned to rank its positive passage above negatives in the
   frozen teacher index.

Data: MS MARCO passages (Tevatron). The index is 299,985 passages, and evaluation uses internal 5K validation and 5K
test query splits. All runs were on 2 × T4 GPUs on Kaggle.

- Write-up, results and notebook links: [blog.md](blog.md)
- Code docs (design, API, config tables): [doc.md](doc.md)
