import modal

app = modal.App("tag-remaining-titles")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "pandas", "pyarrow", "duckdb"
)

titles_vol = modal.Volume.from_name("titles")
kaggle_vol = modal.Volume.from_name("kaggle")

SPACE_TOKEN = "[unused1]"
tag_map = {
    "<CO>": 0, "<CA>": 1, "<K>": 2, "<I>": 3, "<N>": 4,
    "<V>": 5, "<D>": 6, "<M>": 7, "<S>": 8, "<O>": 9,
}
id2tag = {v: k for k, v in tag_map.items()}

src_path = "/mnt/kaggle/classical_tracks_with_roles.parquet"
done_path = "/mnt/titles/tagged_track_titles.parquet"
shard_dir = "/mnt/titles/tagged_remaining_shards"
out_path = "/mnt/titles/tagged_track_titles_remaining.parquet"
model_dir = "/mnt/titles/fine_tuned_ner_model"

batch_size = 256
shard_size = 50_000


def pred_to_text(tokenizer, input_ids, pred_ids):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    cleaned_tokens, cleaned_pred_tags = [], []
    for tok, label_idx in zip(tokens, pred_ids):
        if tok == SPACE_TOKEN:
            tok = " "
        if tok in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]:
            continue
        cleaned_tokens.append(tok)
        cleaned_pred_tags.append(id2tag[label_idx])

    pred_str, current_tag = "", None
    for tok, tag in zip(cleaned_tokens, cleaned_pred_tags):
        text_chunk = tok.replace("##", "")
        if tag != current_tag:
            pred_str += tag + text_chunk
            current_tag = tag
        else:
            pred_str += text_chunk
    return pred_str


def work_list():
    """untagged titles, ordered deterministically so shard numbering is stable across reruns"""
    import duckdb

    con = duckdb.connect()
    return con.execute(f"""
        SELECT DISTINCT s.track_rowid, s.name_track
        FROM read_parquet('{src_path}') s
        ANTI JOIN read_parquet('{done_path}') d
            ON s.track_rowid = d.track_rowid
        ORDER BY s.track_rowid, s.name_track
    """).fetchall()


def load_model():
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tokenizer, model, device


def tag_batch(tokenizer, model, device, titles):
    import torch

    texts = [t.replace(" ", SPACE_TOKEN) for t in titles]
    inputs = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        predictions = torch.argmax(model(**inputs).logits, dim=2)

    return [
        pred_to_text(
            tokenizer,
            inputs["input_ids"][j].cpu().tolist(),
            predictions[j].cpu().tolist(),
        )
        for j in range(len(texts))
    ]


def shard_rows(path):
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


@app.function(
    image=image,
    volumes={"/mnt/titles": titles_vol, "/mnt/kaggle": kaggle_vol},
    gpu="T4",
    timeout=1800,
)
def smoke():
    import time
    import duckdb

    t0 = time.time()
    work = work_list()
    print(f"untagged titles: {len(work):,} (query {time.time() - t0:.1f}s)", flush=True)

    con = duckdb.connect()
    print("existing file schema:", con.execute(f"DESCRIBE SELECT * FROM read_parquet('{done_path}')").fetchall())

    tokenizer, model, device = load_model()
    sample = work[:batch_size]
    tagged = tag_batch(tokenizer, model, device, [t if t is not None else "" for _, t in sample])

    for (rowid, title), out in list(zip(sample, tagged))[:5]:
        print(f"{rowid} | {title!r} -> {out!r}")
    print("tagged>0 chars:", sum(1 for x in tagged if x), "/", len(tagged))

    return len(work)


@app.function(
    image=image,
    volumes={"/mnt/titles": titles_vol, "/mnt/kaggle": kaggle_vol},
    gpu="T4",
    timeout=21600,
)
def run():
    import glob
    import os
    import time

    import duckdb
    import pandas as pd

    os.makedirs(shard_dir, exist_ok=True)
    t0 = time.time()

    work = work_list()
    n = len(work)
    print(f"untagged titles: {n:,}", flush=True)

    tokenizer = model = device = None
    for shard_idx, start in enumerate(range(0, n, shard_size)):
        path = f"{shard_dir}/shard_{shard_idx:05d}.parquet"
        chunk = work[start:start + shard_size]

        if os.path.exists(path) and shard_rows(path) == len(chunk):
            print(f"skip shard {shard_idx:05d} (already done, {len(chunk):,} titles)", flush=True)
            continue

        if tokenizer is None:
            tokenizer, model, device = load_model()
            print(f"model loaded on {device} after {time.time() - t0:.1f}s", flush=True)

        frames, predicted = [], 0
        for i in range(0, len(chunk), batch_size):
            batch = chunk[i:i + batch_size]
            ids = [r[0] for r in batch]
            titles = [r[1] if r[1] is not None else "" for r in batch]
            tagged = tag_batch(tokenizer, model, device, titles)
            frames.append(pd.DataFrame({"track_rowid": ids, "tagged_name_track": tagged}))
            predicted += len(batch)

            if predicted % (batch_size * 50) < batch_size or predicted >= len(chunk):
                done = start + predicted
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (n - done) / rate if rate else 0
                print(
                    f"{done:,}/{n:,} | {elapsed / 60:.1f}min | {rate:.0f} titles/s "
                    f"| eta {eta / 60:.1f}min",
                    flush=True,
                )

        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(path, index=False)
        titles_vol.commit()
        print(f"wrote {path} ({len(df):,} titles)", flush=True)

    shards = sorted(glob.glob(f"{shard_dir}/shard_*.parquet"))
    shard_list = "[" + ", ".join(f"'{s}'" for s in shards) + "]"
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT track_rowid, tagged_name_track
            FROM read_parquet({shard_list})
        ) TO '{out_path}' (FORMAT PARQUET)
    """)
    titles_vol.commit()

    total = con.execute(f"SELECT count(*) FROM read_parquet('{out_path}')").fetchone()[0]
    print(f"consolidated {len(shards)} shards -> {out_path}: {total:,} rows", flush=True)
    return total


if __name__ == "__main__":
    with app.run():
        print(run.remote())
