"""
inspect_hf_schema.py — One-time diagnostic script.

The Voxel51/mvtec-ad Hugging Face mirror is a FiftyOne dataset, not a plain
imagefolder. It's known to expose metadata like `category.label` and
`defect.label` (confirmed from Voxel51's own docs), but the exact field
name used for the pixel-level defect *mask* (a fo.Segmentation field) and
whether there's an explicit train/test split field isn't documented
anywhere public — so rather than guess and risk building dataset.py
against the wrong field names, this script loads the dataset once and
prints its real schema + a sample record.

Run this once, then send me the output and I'll build the real
HuggingFace-backed Dataset class against the confirmed field names.
"""

import fiftyone as fo
import fiftyone.utils.huggingface as fouh

print("Downloading / loading Voxel51/mvtec-ad via FiftyOne (first run will "
      "download ~5.2GB, cached afterward)...")

dataset = fouh.load_from_hub("Voxel51/mvtec-ad", persistent=True, overwrite=True)

print("\n=== Dataset summary ===")
print(dataset)

print("\n=== Field schema ===")
for field_name, field in dataset.get_field_schema().items():
    print(f"  {field_name}: {field}")

print("\n=== First sample (full field dump) ===")
first = dataset.first()
print(first)

print("\n=== Distinct values for likely categorical fields ===")
for candidate in ["category.label", "defect.label", "split", "tags"]:
    try:
        values = dataset.distinct(candidate)
        print(f"  {candidate}: {values[:20]}{' ...' if len(values) > 20 else ''}")
    except Exception as e:
        print(f"  {candidate}: (not found / error: {e})")

print("\nDone. Copy everything above and send it back.")