"""
dataset.py — MVTec AD dataset loader.

Two backends, same output interface:

1. MVTecDataset — reads the original MVTec folder layout directly
   (<root>/<category>/train/good/*.png, etc). Useful if you download
   the dataset from the official MVTec site.

2. MVTecHFDataset — reads the Voxel51/mvtec-ad FiftyOne/HuggingFace
   mirror (schema confirmed by running inspect_hf_schema.py):
       filepath:    path to the cached image on disk
       category.label: one of the 15 MVTec object categories
       defect.label:   "good" for normal, else the defect type name
       split:          "train" or "test"
       defect_mask:    fo.Segmentation, None for normal images,
                        a per-pixel mask array for defective ones

Both return (image, mask) pairs ready for PhysicsConstrainedCNN /
PhysicsConstrainedLoss: image is (3, H, W) float, normalized,
mask is (1, H, W) float in {0, 1}.

Run this file directly to build a small synthetic MVTec-like folder
structure in a temp dir and verify the folder-based loader without
requiring the real (5GB) dataset. Run with --hf to instead test against
the already-downloaded HuggingFace/FiftyOne mirror.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

def augment_pair(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Stronger paired augmentation for small datasets (~200 images/category).

    Geometric transforms (flip, rotation, translation, scale) are applied
    identically to both image and mask so they stay pixel-aligned. Color
    jitter is applied to the image only, since defect color/contrast
    should vary but the mask's geometry doesn't depend on color.

    Mask uses NEAREST interpolation throughout to preserve hard {0, 1}
    boundaries instead of introducing blurred/fractional edge values.
    """
    if np.random.rand() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    if np.random.rand() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    angle = float(np.random.uniform(-15, 15))
    translate = (
        int(np.random.uniform(-0.05, 0.05) * image.size[0]),
        int(np.random.uniform(-0.05, 0.05) * image.size[1]),
    )
    scale = float(np.random.uniform(0.9, 1.1))
    image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=0,
                       interpolation=transforms.InterpolationMode.BILINEAR)
    mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=0,
                      interpolation=transforms.InterpolationMode.NEAREST)

    # Color jitter: image only, mask geometry is unaffected by color.
    if np.random.rand() > 0.3:
        brightness = float(np.random.uniform(0.8, 1.2))
        contrast = float(np.random.uniform(0.8, 1.2))
        saturation = float(np.random.uniform(0.8, 1.2))
        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        image = TF.adjust_saturation(image, saturation)

    return image, mask

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    """
    Args:
        root: path to the MVTec AD root directory (contains one folder
            per category, e.g. root/bottle, root/cable, ...).
        category: one of MVTEC_CATEGORIES, or "all" to pool every category.
        split: "train" (normal images only, as MVTec provides) or "test"
            (normal + anomalous images with masks).
        image_size: images and masks are resized to (image_size, image_size).
        augment: apply light train-time augmentation (flips). Only used
            when split == "train".
    """

    def __init__(
        self,
        root: str | Path,
        category: str = "bottle",
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
    ):
        assert split in ("train", "test")
        self.root = Path(root)
        self.image_size = image_size
        self.split = split
        self.augment = augment and split == "train"

        categories = MVTEC_CATEGORIES if category == "all" else [category]
        self.samples: list[tuple[Path, Path | None]] = []  # (image_path, mask_path_or_None)

        for cat in categories:
            cat_dir = self.root / cat
            if not cat_dir.is_dir():
                continue
            if split == "train":
                for img_path in sorted((cat_dir / "train" / "good").glob("*.png")):
                    self.samples.append((img_path, None))
            else:
                test_dir = cat_dir / "test"
                for defect_dir in sorted(test_dir.iterdir()):
                    if not defect_dir.is_dir():
                        continue
                    for img_path in sorted(defect_dir.glob("*.png")):
                        if defect_dir.name == "good":
                            self.samples.append((img_path, None))
                        else:
                            mask_path = (
                                cat_dir / "ground_truth" / defect_dir.name
                                / f"{img_path.stem}_mask.png"
                            )
                            self.samples.append(
                                (img_path, mask_path if mask_path.exists() else None)
                            )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found under {self.root} for category={category!r}, "
                f"split={split!r}. Check the MVTec AD directory layout."
            )

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.mask_resize = transforms.Resize(
            (image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", image.size, 0)  # all-zero: no defect

        if self.augment:
            image, mask = augment_pair(image, mask)

        image_t = self.image_transform(image)

        mask = self.mask_resize(mask)
        mask_t = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0).unsqueeze(0)
        mask_t = (mask_t > 0.5).float()

        return {
            "image": image_t,
            "mask": mask_t,
            "path": str(img_path),
            "has_defect": mask_t.sum().item() > 0,
        }


class MVTecHFDataset(Dataset):
    """
    MVTec AD loader backed by the Voxel51/mvtec-ad FiftyOne/HuggingFace
    mirror. Assumes the dataset has already been downloaded once via
    fiftyone.utils.huggingface.load_from_hub (e.g. by running
    inspect_hf_schema.py) — this class just opens the existing local
    FiftyOne dataset by name, it does not re-download anything.

    IMPORTANT — split semantics differ from MVTec's official protocol:
    MVTec AD's official `train` split contains ONLY normal (defect-free)
    images, by design, for unsupervised anomaly detection. A supervised
    segmentation model (like PhysicsConstrainedCNN, trained against real
    per-pixel defect masks) would never see a single positive example
    under that protocol — it degenerately learns to always predict
    "background everywhere," which fits the all-normal training targets
    perfectly but fails on real defects at validation time.

    To fix this, by default (resplit_defects=True) this loader builds a
    custom split instead: normal images + a fraction of labeled defective
    images (with real masks) go into "train"; the remaining defective
    images + normal test images are held out for "test"/validation. The
    defect-image split is deterministic (fixed seed) and stratified per
    category so results are reproducible.

    Args:
        category: one of MVTEC_CATEGORIES, or "all" to pool every category.
        split: "train" or "test".
        image_size: images and masks are resized to (image_size, image_size).
        augment: apply augmentation. Only used when split == "train".
        resplit_defects: if True (default, recommended for this project),
            use the custom re-split described above. If False, use
            MVTec's official split field directly (train = all normal,
            test = normal + defective) — NOT recommended for supervised
            mask-based training, since train would then contain zero
            positive examples.
        defect_train_fraction: fraction of labeled defective images
            (per category) assigned to train under the custom re-split.
        seed: fixes the deterministic shuffle used to split defective
            images between train/test, for reproducibility.
    """

    HF_DATASET_NAME = "Voxel51/mvtec-ad"

    def __init__(
        self,
        category: str = "bottle",
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        resplit_defects: bool = True,
        defect_train_fraction: float = 0.6,
        seed: int = 42,
    ):
        assert split in ("train", "test")
        import random

        import fiftyone as fo

        self.image_size = image_size
        self.split = split
        self.augment = augment and split == "train"

        fo_dataset = fo.load_dataset(self.HF_DATASET_NAME)

        category_view = fo_dataset
        if category != "all":
            category_view = category_view.match(fo.ViewField("category.label") == category)

        if resplit_defects:
            normal_train_ids = [
                s.id for s in category_view.match(fo.ViewField("split") == "train")
            ]
            defective_ids = [
                s.id for s in category_view.match(fo.ViewField("defect.label") != "good")
            ]
            good_test_ids = [
                s.id for s in category_view.match(
                    (fo.ViewField("split") == "test") & (fo.ViewField("defect.label") == "good")
                )
            ]

            # Deterministic, reproducible split of defective images.
            defective_ids = sorted(defective_ids)
            rng = random.Random(seed)
            rng.shuffle(defective_ids)
            n_train_defect = int(len(defective_ids) * defect_train_fraction)
            train_defect_ids = defective_ids[:n_train_defect]
            val_defect_ids = defective_ids[n_train_defect:]

            if split == "train":
                final_ids = normal_train_ids + train_defect_ids
            else:
                final_ids = good_test_ids + val_defect_ids

            view = fo_dataset.select(final_ids)
        else:
            view = category_view.match(fo.ViewField("split") == split)

        # Materialize the fields we need up front; keeps __getitem__ cheap
        # and avoids repeated FiftyOne queries during training.
        self.samples: list[dict] = []
        for sample in view.select_fields(
            ["filepath", "category", "defect", "defect_mask"]
        ):
            mask_path = None
            if sample.defect_mask is not None:
                mask_path = sample.defect_mask.mask_path or None
            self.samples.append({
                "filepath": sample.filepath,
                "category": sample.category.label if sample.category else None,
                "defect": sample.defect.label if sample.defect else "good",
                "mask_path": mask_path,
            })

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found in '{self.HF_DATASET_NAME}' for "
                f"category={category!r}, split={split!r}. Did "
                f"inspect_hf_schema.py finish successfully?"
            )

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.mask_resize = transforms.Resize(
            (image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.samples[idx]

        image = Image.open(record["filepath"]).convert("RGB")

        mask_path = record["mask_path"]
        if mask_path is not None:
            # Mask PNGs from MVTec are typically 0 = background,
            # 255 (or any nonzero) = defect region.
            raw_mask = Image.open(mask_path).convert("L")
            mask = Image.fromarray(
                (np.asarray(raw_mask) > 0).astype(np.uint8) * 255, mode="L"
            )
        else:
            mask = Image.new("L", image.size, 0)  # normal image: no defect anywhere

        if self.augment:
            image, mask = augment_pair(image, mask)

        image_t = self.image_transform(image)

        mask = self.mask_resize(mask)
        mask_t = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0).unsqueeze(0)
        mask_t = (mask_t > 0.5).float()

        return {
            "image": image_t,
            "mask": mask_t,
            "path": record["filepath"],
            "category": record["category"],
            "defect": record["defect"],
            "has_defect": mask_t.sum().item() > 0,
        }


def _build_synthetic_mvtec(tmp_root: Path, category: str = "bottle") -> None:
    """Builds a tiny fake MVTec-style folder tree for self-testing."""
    cat_dir = tmp_root / category
    (cat_dir / "train" / "good").mkdir(parents=True)
    (cat_dir / "test" / "good").mkdir(parents=True)
    (cat_dir / "test" / "crack").mkdir(parents=True)
    (cat_dir / "ground_truth" / "crack").mkdir(parents=True)

    rng = np.random.default_rng(0)

    for i in range(4):
        img = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
        img.save(cat_dir / "train" / "good" / f"{i:03d}.png")

    for i in range(2):
        img = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
        img.save(cat_dir / "test" / "good" / f"{i:03d}.png")

    for i in range(2):
        img = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
        img.save(cat_dir / "test" / "crack" / f"{i:03d}.png")
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[40:80, 40:80] = 255  # synthetic defect region
        Image.fromarray(mask).save(cat_dir / "ground_truth" / "crack" / f"{i:03d}_mask.png")


def _test_hf_backend() -> None:
    from torch.utils.data import DataLoader

    print("Testing MVTecHFDataset (custom re-split) against the real downloaded dataset...")
    train_ds = MVTecHFDataset(category="bottle", split="train", image_size=256, resplit_defects=True)
    test_ds = MVTecHFDataset(category="bottle", split="test", image_size=256, resplit_defects=True)

    print(f"\nTrain samples (bottle): {len(train_ds)}")
    print(f"Test samples  (bottle): {len(test_ds)}")

    # The whole point of the re-split: training data must contain real
    # positive (defective) examples, unlike MVTec's official train split.
    train_defect_count = sum(1 for s in train_ds.samples if s["mask_path"] is not None)
    test_defect_count = sum(1 for s in test_ds.samples if s["mask_path"] is not None)
    print(f"\nTrain samples WITH a defect mask: {train_defect_count} (expected > 0 -- this is the fix)")
    print(f"Test samples WITH a defect mask:  {test_defect_count} (expected > 0 -- held-out defects)")
    assert train_defect_count > 0, "Re-split failed: training set still has zero defective examples."
    assert test_defect_count > 0, "Re-split failed: no defective examples held out for validation."

    train_sample = train_ds[0]
    print(f"\nTrain sample image shape: {tuple(train_sample['image'].shape)}")
    print(f"Train sample mask shape:  {tuple(train_sample['mask'].shape)}")

    defect_found_in_train_batch = False
    for sample in train_ds:
        if sample["has_defect"]:
            defect_found_in_train_batch = True
            assert sample["mask"].sum() > 0
            break
    print(f"\nFound at least one defective TRAINING sample with nonzero mask: {defect_found_in_train_batch}")
    assert defect_found_in_train_batch

    loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"\nBatch image shape: {tuple(batch['image'].shape)}")
    print(f"Batch mask shape:  {tuple(batch['mask'].shape)}")
    assert batch["image"].shape == (4, 3, 256, 256)
    assert batch["mask"].shape == (4, 1, 256, 256)

    print("\nAll HF-backend re-split checks passed.")


if __name__ == "__main__":
    import sys

    if "--hf" in sys.argv:
        _test_hf_backend()
        raise SystemExit(0)

    tmp_dir = Path(tempfile.mkdtemp(prefix="mvtec_synthetic_"))
    try:
        print(f"Building synthetic MVTec-style dataset at {tmp_dir} ...")
        _build_synthetic_mvtec(tmp_dir, category="bottle")

        train_ds = MVTecDataset(tmp_dir, category="bottle", split="train", image_size=256)
        test_ds = MVTecDataset(tmp_dir, category="bottle", split="test", image_size=256)

        print(f"\nTrain samples: {len(train_ds)} (expected 4, all normal)")
        print(f"Test samples:  {len(test_ds)} (expected 4: 2 good + 2 crack)")
        assert len(train_ds) == 4
        assert len(test_ds) == 4

        train_sample = train_ds[0]
        print(f"\nTrain sample image shape: {tuple(train_sample['image'].shape)}")
        print(f"Train sample mask shape:  {tuple(train_sample['mask'].shape)}")
        print(f"Train sample has_defect:  {train_sample['has_defect']} (expected False)")
        assert train_sample["image"].shape == (3, 256, 256)
        assert train_sample["mask"].shape == (1, 256, 256)
        assert train_sample["has_defect"] is False

        defect_found = False
        good_found = False
        for sample in test_ds:
            if sample["has_defect"]:
                defect_found = True
                assert sample["mask"].sum() > 0
            else:
                good_found = True
                assert sample["mask"].sum() == 0
        print(f"\nFound at least one defective test sample: {defect_found} (expected True)")
        print(f"Found at least one normal test sample:     {good_found} (expected True)")
        assert defect_found and good_found

        # Check batching via DataLoader works end to end.
        from torch.utils.data import DataLoader
        loader = DataLoader(test_ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        print(f"\nBatch image shape: {tuple(batch['image'].shape)}")
        print(f"Batch mask shape:  {tuple(batch['mask'].shape)}")
        assert batch["image"].shape == (2, 3, 256, 256)
        assert batch["mask"].shape == (2, 1, 256, 256)

        print("\nAll checks passed.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)