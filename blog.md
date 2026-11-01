# Progressive Embedding Distillation

### A Three-Phase Training Recipe for Building Efficient Retrieval Models

## 1. Problem Statement

Large embedding models provide strong retrieval performance but are expensive to deploy, particularly when query encoding needs to happen online with low latency.

The objective of this project is to investigate whether a smaller embedding model can approximate the representation space of a strong teacher model while retaining retrieval performance.

I explore a three-phase training recipe:

1. **L2 Alignment:** Align a smaller student model with the teacher's embedding representations.
2. **Structured Pruning and Recovery:** Progressively reduce student model depth and recover the pruned model using L2 alignment.
3. **Contrastive Refinement:** Improve retrieval-specific discrimination between relevant and irrelevant documents.

The overall goal is to reduce model size and inference latency while retaining compatibility with a fixed teacher document index.

## 2. Teacher Embedding Generation

I use a Qwen 4B embedding model as the teacher. The teacher is kept frozen, and its embeddings are precomputed for queries and documents.

I did not fine-tune the teacher itself. It already provided a strong retrieval baseline, and the available computational budget was insufficient to justify additional teacher training. The objective was to transfer its embedding representations to smaller models rather than improve the teacher.

All embeddings are generated with an output dimension of **256** and cached for subsequent experiments.

**Notebook:** [Teacher Embedding Generation](https://www.kaggle.com/code/tensorpanda231/01-generate-teacher-embeddings)

## 3. Dataset and Experimental Setup

I use the [Tevatron MS MARCO Passage dataset](https://huggingface.co/datasets/Tevatron/msmarco-passage/viewer/default/train?row=0).

The dataset contains queries along with their corresponding positive and negative passages. I selected a subset because it provides useful retrieval supervision while remaining manageable with the available computational resources.

The experimental data is divided as follows:

| Split            | Records                                  | Usage                                |
| ---------------- | ---------------------------------------- | ------------------------------------ |
| Validation       | 5K queries with passages                 | Validation                           |
| Test             | 5K queries with passages                 | Final evaluation                     |
| Train            | 90K queries with positives and negatives | Phases 1, 2 and 3                    |
| Additional train | 110K queries, without passages           | Phases 1 and 2 (L2 alignment only)   |

The retrieval corpus consists of **299,985 passages**: the positives and one stored negative of each of the 100K core (train, validation and test) queries, about 200K passages, plus 100K random distractor passages.

An important detail is that distractor documents are part of the fixed teacher document index. The student does not receive these documents as alignment or training targets. They serve exclusively as competing candidates during retrieval.

### Computational Setup

All experiments were conducted using two NVIDIA T4 GPUs on Kaggle.

| Configuration                    | Value                                        |
| -------------------------------- | -------------------------------------------- |
| Hardware                         | 2 × NVIDIA T4                                |
| Teacher output dimension         | 256                                          |
| Teacher query maximum length     | 64 (including the ~20-token instruction prompt) |
| Student query maximum length     | 32                                           |
| Document maximum sequence length | 128                                          |
| Retrieval corpus                 | 299,985 passages                             |

## 4. Teacher Baseline

Before training the student models, I evaluated the teacher against the retrieval corpus on the internal test split (5K queries).

| Metric      | Teacher |
| ----------- | ------- |
| Recall\@100 | 0.9928  |
| Recall\@10  | 0.9457  |
| nDCG\@10    | 0.8063  |
| MRR\@10     | 0.7639  |

These metrics capture different aspects of retrieval quality:

- **Recall\@K:** Measures the fraction of relevant documents retrieved within the top K results.

  `Recall@K = |Relevant ∩ TopK| / |Relevant|`
- **nDCG\@10:** Measures ranking quality by accounting for both relevance and position. It normalizes discounted cumulative gain against the ideal ranking.

  `DCG@10 = Σ [(2^rel_i - 1) / log2(i + 1)]`

  `nDCG@10 = DCG@10 / IDCG@10`
- **MRR\@10:** Measures the reciprocal rank of the first relevant document, averaged across queries.

  `MRR@10 = (1 / |Q|) Σ [1 / rank_q]`

  Queries without a relevant result in the top 10 receive a score of zero.

Recall measures retrieval coverage, nDCG measures the quality of the overall ranking, and MRR emphasizes how quickly the first relevant result appears.

These metrics together provide a more complete picture than using recall alone.

## 5. Three-Phase Training Recipe

### Phase 1: L2 Alignment

I use Qwen 0.6B as the initial student model.

The objective is to train the student to reproduce the teacher's embeddings for the same queries. Only queries are aligned: the student is a query encoder, and documents are embedded once by the teacher and never encoded by the student. Teacher embeddings are already cached, so the teacher does not need to participate in the student training loop.

This phase attempts to transfer the teacher's embedding space to a smaller model.

### Phase 2: Structured Pruning and Recovery

After obtaining an aligned student, I investigate reducing its depth through structured layer pruning.

My initial pruning strategy removes evenly spaced Transformer layers. The remaining layers are then trained again using the L2 alignment objective to recover the lost representation quality.

I also conducted additional experiments with importance-based pruning to investigate whether selecting layers based on their importance produces better results than uniform depth reduction.

**Notebook:** [Student Alignment and Pruning](https://www.kaggle.com/code/tensorpanda231/full-finetune-02-student-alignment-and-pruning)

### Phase 3: Contrastive Refinement

L2 alignment encourages the student to reproduce teacher embeddings, but vector approximation and retrieval discrimination are not necessarily the same objective.

I therefore introduce contrastive refinement using the available positive and negative passages.

The student is trained to produce query representations that rank relevant passages above negative passages, using the fixed teacher document index.

The motivation is to complement representation-level alignment with task-specific retrieval supervision.

**Notebook:** [Contrastive Refinement](https://www.kaggle.com/code/tensorpanda231/v3-03-contrastive-refinement)

## 6. Student Architecture Experiments

An important part of this investigation was understanding whether the proposed approach depends on a particular student architecture.

I conducted two additional sets of experiments:

**Different embedding architectures:** I evaluated several independent embedding models in the 100–150M parameter range. The objective was to determine whether the alignment and refinement recipe generalizes beyond the Qwen architecture.

[Notebook: Strong Student Models](https://www.kaggle.com/code/tensorpanda231/06-strong-students-full-finetune)

**Different model capacities:** I also experimented with student models from the same model family, with progressively increasing parameter counts.

These experiments help investigate the capacity limitations of L2 alignment. Smaller models may have limited ability to approximate the teacher's embedding representations, whereas larger students can potentially reproduce them more accurately.

Contrastive refinement provides an additional retrieval-specific learning signal, helping the student distinguish positive and negative documents rather than focusing exclusively on matching teacher vectors.

## 7. Retrieval and Latency Evaluation

The final evaluation compares the teacher, aligned students, pruned students, and contrastively refined models.

Along with retrieval metrics, I investigate inference latency to understand the practical benefits of model compression.

[Notebook: Retrieval Evaluation and Ablations](https://www.kaggle.com/code/tensorpanda231/04-retrieval-evaluation-and-ablation)

The main evaluation criteria are:

- Retrieval effectiveness: Recall\@10, Recall\@100, nDCG\@10 and MRR\@10.
- Model efficiency: Parameter count and inference latency.
- Compression effectiveness: Retrieval performance retained after pruning and recovery.

## 8. Future Directions

There are several directions worth exploring further:

1. **Quantization:** Evaluate INT8 and other quantization strategies to reduce memory consumption and improve online inference latency.
2. **Pruning strategies:** Investigate better importance-based and retrieval-aware structured pruning techniques.
3. **Contrastive learning:** Explore alternative contrastive objectives, hard-negative strategies and training configurations.
4. **Embedding alignment:** Investigate the limits of L2-based alignment across different model capacities and architectures.

An interesting related work is [LEAF: Knowledge Distillation of Text Embedding Models with Teacher-Aligned Representations (ACL 2026)](https://aclanthology.org/2026.acl-long.2008.pdf).

LEAF demonstrates that directly aligning student representations with teacher embeddings using an L2 objective can be an effective knowledge distillation strategy. Its experiments also report that adding more elaborate internal representation losses did not improve its downstream retrieval results.

This provides further motivation to investigate embedding alignment as a standalone foundation and study how pruning, recovery and contrastive refinement can complement it.

## 9. Conclusion

This project investigates a progressive approach to building efficient retrieval models by combining teacher embedding alignment, structured model pruning, recovery and contrastive refinement.

Rather than treating compression as an isolated operation, the objective is to study how representation learning and task-specific optimization can work together to produce smaller retrieval models with competitive effectiveness and better inference efficiency.

Further experimental results and ablation studies will help establish the effectiveness and limitations of this approach.
