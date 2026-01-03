## Team Testing (Low-Code)

This project supports **arbitrary “data dump” folder structures** for team testing. You can point the system at any directory (labeled or unlabeled) and build an embeddings + knowledge-graph artifact for repeatable querying.

### 1) Build artifacts from any data folders

Example using your existing folders:

```bash
python scripts/gcs_cli.py build \
  --data-root data \
  --data-root balanced_data \
  --artifact artifacts/gcs_artifacts.pkl
```

How folder structure is interpreted:
- `--dataset-level` (default `0`): which relative folder segment is treated as “dataset”
- `--label-level` (default `1`): which relative folder segment is treated as “label” (use `none` for unlabeled dumps)

For example, with `balanced_data/balanced_parkinson/normal/img.png` and defaults:
- dataset = `balanced_parkinson` (segment 0)
- label = `normal` (segment 1)

If your dump has no labels:

```bash
python scripts/gcs_cli.py build \
  --data-root /path/to/any_dump \
  --label-level none \
  --artifact artifacts/gcs_artifacts.pkl
```

### 2) Query (text, image, or both)

Text query:
```bash
python scripts/gcs_cli.py query --artifact artifacts/gcs_artifacts.pkl --text "MRI with lesions" --mode auto --top-k 10
```

Image query:
```bash
python scripts/gcs_cli.py query --artifact artifacts/gcs_artifacts.pkl --image /path/to/query.jpg --mode local --top-k 10
```

Multimodal (text + image): the CLI fuses both embeddings.
```bash
python scripts/gcs_cli.py query --artifact artifacts/gcs_artifacts.pkl --text "similar scans" --image /path/to/query.jpg --mode hybrid --top-k 10
```

### 3) Single-shot tagging (propagate in embedding space)

Assign a tag to the *k* nearest neighbors of a seed image:
```bash
python scripts/gcs_cli.py tag --artifact artifacts/gcs_artifacts.pkl --image /path/to/seed.jpg --tag "review_me" --k 50
```

This updates the saved artifact by adding `user_tags` metadata to the nearest images (use `--out` to write a new artifact file).

