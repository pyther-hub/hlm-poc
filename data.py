"""MS MARCO (Tevatron/msmarco-passage, train split) → by-query splits, corpus, alignment queries, CR triples.
Passages are referred to everywhere by their row in the corpus table (data["corpus"]["ids"][i]).
Also holds the small I/O helpers for the cached artifacts (data.json, embedding .npy files, cfg merging)."""
import json
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from model import tokenize

DATASET = "Tevatron/msmarco-passage"


# ---------- rows ----------

def slim_row(row):
    """Keep only what we use. negative_passages[0] is the one stored negative
    (Tevatron-provided; README says its BM25 origin is unverified)."""
    def p(x):
        return {"docid": x["docid"], "title": x["title"], "text": x["text"]}
    return {"query_id": row["query_id"], "query": row["query"],
            "positives": [p(x) for x in row["positive_passages"]], "negative": p(row["negative_passages"][0])}


def clean_rows(raw_rows):
    """slim + drop rows without a positive or negative + dedupe by query_id (so splits are truly by query)."""
    rows, seen = [], set()
    for r in raw_rows:
        if not r["positive_passages"] or not r["negative_passages"] or r["query_id"] in seen:
            continue
        seen.add(r["query_id"])
        rows.append(slim_row(r))
    return rows


def load_rows(n_rows, cache_dir):
    """First n_rows of the train split, streamed (no 1.7 GB download), cached as parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = os.path.join(cache_dir, f"rows_{n_rows}.parquet")
    if not os.path.exists(path):
        from datasets import load_dataset
        from tqdm.auto import tqdm
        stream = load_dataset(DATASET, split="train", streaming=True).take(n_rows)
        rows = clean_rows(tqdm(stream, total=n_rows, desc="msmarco rows"))
        os.makedirs(cache_dir, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path + ".tmp")
        os.replace(path + ".tmp", path)
    return pq.read_table(path).to_pylist()


def tiny_rows(n=60, n_docs=90, seed=0):
    """Fake rows in the raw Tevatron format (then cleaned), with shared and multiple positives."""
    rng = np.random.default_rng(seed)
    words = "river city music cell gold tax moon salt king rain bank code lake film bird wave".split()

    def doc(i):
        return {"docid": f"d{i}", "title": "-", "text": " ".join(rng.choice(words, 12))}
    raw = []
    for q in range(n):
        pos = [doc(int(rng.integers(n_docs))) for _ in range(1 + (q % 3 == 0))]
        raw.append({"query_id": f"q{q}", "query": " ".join(rng.choice(words, 4)) + f" {q}",
                    "positive_passages": pos, "negative_passages": [doc(int(rng.integers(n_docs))) for _ in range(3)]})
    return clean_rows(raw)


# ---------- splits, corpus, query sets ----------

def make_splits(n_rows, core_n, n_val, n_test, split_seed):
    """Row indices. Core = seeded sample of core_n queries → train/val/test; the rest are "other"."""
    perm = np.random.default_rng(split_seed).permutation(n_rows).tolist()
    core, other = perm[:core_n], perm[core_n:]
    return {"val": core[:n_val], "test": core[n_val:n_val + n_test], "train": core[n_val + n_test:], "other": other}


def passage_text(p, use_title):
    title = p["title"].strip()
    return f"{title}. {p['text']}" if use_title and title not in ("", "-") else p["text"]


def build_corpus(rows, core_idx, use_title=False):
    """All positives of the core queries + their stored negative, deduplicated by docid."""
    ids, texts, index = [], [], {}
    for i in core_idx:
        for p in rows[i]["positives"] + [rows[i]["negative"]]:
            if p["docid"] not in index:
                index[p["docid"]] = len(ids)
                ids.append(p["docid"])
                texts.append(passage_text(p, use_title))
    return {"ids": ids, "texts": texts}, index


def query_set(rows, idx, doc_index):
    """ids, texts, pos (list of corpus rows per query), neg (corpus row of the stored negative)."""
    return {"ids": [rows[i]["query_id"] for i in idx],
            "texts": [rows[i]["query"] for i in idx],
            "pos": [[doc_index[p["docid"]] for p in rows[i]["positives"]] for i in idx],
            "neg": [doc_index[rows[i]["negative"]["docid"]] for i in idx]}


def alignment_queries(rows, split, n_align):
    """Train queries + "other" queries up to n_align, never a val/test query (by id or by text)."""
    held_out = {rows[i]["query"].strip().lower() for i in split["val"] + split["test"]}
    idx = list(split["train"])
    for i in split["other"]:
        if len(idx) >= n_align:
            break
        if rows[i]["query"].strip().lower() not in held_out:
            idx.append(i)
    if len(idx) < n_align:
        print(f"[data] only {len(idx)} alignment queries available (asked for {n_align})")
    return {"ids": [rows[i]["query_id"] for i in idx], "texts": [rows[i]["query"] for i in idx]}


def build_data(rows, core_n, n_val, n_test, n_align, split_seed, use_title=False):
    split = make_splits(len(rows), core_n, n_val, n_test, split_seed)
    corpus, doc_index = build_corpus(rows, split["train"] + split["val"] + split["test"], use_title)
    data = {"corpus": corpus, "align": alignment_queries(rows, split, n_align)}
    for name in ("train", "val", "test"):
        data[name] = query_set(rows, split[name], doc_index)
    print(f"[data] corpus={len(corpus['ids'])} train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])} align={len(data['align']['ids'])}")
    return data


def cr_triples(data):
    """(query text, positive corpus row, hard-negative corpus row) for every train query."""
    return [(t, pos[0], neg) for t, pos, neg in zip(data["train"]["texts"], data["train"]["pos"], data["train"]["neg"])]


# ---------- loaders ----------

def collate_queries(batch, tokenizer, max_len):
    """batch: list of dicts with "text" and optionally "target" (teacher vector), "pos", "neg" (corpus rows)."""
    enc = tokenize(tokenizer, [b["text"] for b in batch], max_len)
    out = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    if "target" in batch[0]:
        out["target"] = torch.from_numpy(np.stack([b["target"] for b in batch])).float()
    for key in ("pos", "neg"):
        if key in batch[0]:
            out[key] = torch.tensor([b[key] for b in batch], dtype=torch.long)
    return out


def make_loader(items, tokenizer, batch_size, max_len, shuffle=True, distributed=False, seed=0, drop_last=False):
    """distributed=True adds a DistributedSampler (train_loop calls set_epoch)."""
    sampler = DistributedSampler(items, shuffle=shuffle, seed=seed, drop_last=drop_last) if distributed else None
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle and not distributed, sampler=sampler,
                      drop_last=drop_last, collate_fn=partial(collate_queries, tokenizer=tokenizer, max_len=max_len))


# ---------- cached artifacts ----------

def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_emb(emb_dir, name):
    """(fp16 array [N, d], ids) written by teacher.merge_shards."""
    return np.load(os.path.join(emb_dir, f"{name}.npy")), load_json(os.path.join(emb_dir, f"{name}_ids.json"))


def resolve_cfg(defaults, cfg):
    """defaults <- cfg. Unknown keys are an error (catches typos). Dict values (loss_kwargs) are replaced
    whole, not merged, so switching the loss never carries over another loss's kwargs."""
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise KeyError(f"unknown cfg keys: {sorted(unknown)}")
    return {**defaults, **cfg}
