"""One-time diagnostic: inspect the defect_mask field on a real defective sample."""
import fiftyone as fo

dataset = fo.load_dataset("Voxel51/mvtec-ad")
view = dataset.match(
    (fo.ViewField("category.label") == "bottle")
    & (fo.ViewField("split") == "test")
    & (fo.ViewField("defect.label") != "good")
)

print(f"Defective bottle test samples found: {len(view)}")
sample = view.first()
print("\n=== Full sample dump ===")
print(sample)
print("\n=== defect_mask field detail ===")
dm = sample.defect_mask
print(f"type: {type(dm)}")
print(f"dir:  {[a for a in dir(dm) if not a.startswith('_')]}")
print(f"mask attr:      {getattr(dm, 'mask', 'NO ATTR')}")
print(f"mask_path attr: {getattr(dm, 'mask_path', 'NO ATTR')}")