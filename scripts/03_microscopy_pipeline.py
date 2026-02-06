#!/usr/bin/env python3
"""
Microscopy Nuclei Segmentation Pipeline.

Loads BBBC039 images, segments nuclei with StarDist (or watershed fallback),
extracts features, and generates visualizations.

Usage:
    python scripts/03_microscopy_pipeline.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_images(images_dir, subset_size=None):
    """Load TIFF images from BBBC039."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []
    try:
        import tifffile
    except ImportError:
        import imageio

        tifffile = imageio

    paths = sorted(images_dir.glob("*.tif*"))[: (subset_size or 999)]
    images = []
    for p in paths:
        try:
            img = tifffile.imread(str(p))
            images.append((p.name, img))
        except Exception as e:
            print(f"  Skip {p.name}: {e}")
    return images


def segment_stardist(img, model_name="2D_versatile_he"):
    """Segment nuclei using StarDist."""
    try:
        from stardist.models import StarDist2D

        model = StarDist2D.from_pretrained(model_name)
        labels, _ = model.predict_instances(img)
        return labels
    except Exception as e:
        print(f"  StarDist failed: {e}")
        return None


def segment_watershed(img):
    """Fallback: simple watershed-based segmentation."""
    from skimage import exposure, feature, filters, measure, morphology, segmentation

    # Ensure 2D
    if img.ndim > 2:
        img = img.squeeze()
        if img.ndim > 2:
            img = img[0]
    # Normalize
    p2, p98 = np.percentile(img, (2, 98))
    img_norm = exposure.rescale_intensity(img, in_range=(p2, p98))
    # Threshold
    thresh = filters.threshold_otsu(img_norm)
    binary = img_norm > thresh
    binary = morphology.remove_small_objects(binary, min_size=50)
    binary = morphology.remove_small_holes(binary, area_threshold=50)
    # Distance + watershed
    from scipy import ndimage

    dist = ndimage.distance_transform_edt(binary)
    coords = feature.peak_local_max(dist, min_distance=5, labels=binary)
    mask = np.zeros_like(dist, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = measure.label(mask)
    labels = segmentation.watershed(-dist, markers, mask=binary)
    return labels


def extract_features(labels):
    """Extract per-nucleus features from label image."""
    from skimage import measure

    if labels is None or labels.max() == 0:
        return pd.DataFrame()

    regions = measure.regionprops(labels)
    rows = []
    for r in regions:
        area = r.area
        perimeter = r.perimeter
        circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
        rows.append(
            {
                "area": area,
                "perimeter": perimeter,
                "circularity": min(circularity, 1.0),
            }
        )
    return pd.DataFrame(rows)


def run_pipeline(config):
    microscopy_dir = Path(config["paths"]["microscopy_dir"]) / "images"
    output_dir = Path(config["paths"]["output_dir"]) / "microscopy"
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = config["microscopy"].get("subset_size") or 20
    images = load_images(microscopy_dir, subset_size=subset)

    if not images:
        print(f"No images in {microscopy_dir}")
        return None

    print(f"Processing {len(images)} images...")
    all_features = []
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    use_stardist = True
    for name, img in images:
        try:
            if use_stardist:
                labels = segment_stardist(img, config["microscopy"]["stardist_model"])
                if labels is None:
                    use_stardist = False
                    labels = segment_watershed(img)
            else:
                labels = segment_watershed(img)

            feats = extract_features(labels)
            feats["image"] = name
            all_features.append(feats)

            # Save overlay
            try:
                from skimage import color

                overlay = color.label2rgb(labels, img, alpha=0.4, bg_label=0)
                from matplotlib import pyplot as plt

                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax[0].imshow(img, cmap="gray")
                ax[0].set_title("Original")
                ax[0].axis("off")
                ax[1].imshow(overlay)
                ax[1].set_title(f"Nuclei ({labels.max()} detected)")
                ax[1].axis("off")
                fig.savefig(overlay_dir / name.replace(".tif", ".png"), dpi=100)
                plt.close()
            except Exception:
                pass

        except Exception as e:
            print(f"  Error {name}: {e}")

    if all_features:
        df = pd.concat(all_features, ignore_index=True)
        df.to_csv(output_dir / "nuclei_features.csv", index=False)
        summary = (
            df.groupby("image")
            .agg(
                area_mean=("area", "mean"),
                area_std=("area", "std"),
                circularity_mean=("circularity", "mean"),
                n_nuclei=("area", "count"),
            )
            .round(2)
        )
        summary.to_csv(output_dir / "summary_stats.csv")
        print(f"Outputs saved to {output_dir}")
    return output_dir


def main():
    config = load_config()
    os.chdir(ROOT)
    run_pipeline(config)


if __name__ == "__main__":
    main()
