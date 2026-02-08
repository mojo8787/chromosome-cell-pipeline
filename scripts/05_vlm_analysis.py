#!/usr/bin/env python3
"""
VLM Analysis Pipeline.

Processes microscopy overlay images through a Vision-Language Model API,
extracts phenotype descriptions, and generates embeddings for downstream
correlation with genomic data.

Usage:
    python scripts/05_vlm_analysis.py

Requires: OPENAI_API_KEY (and ANTHROPIC_API_KEY if backend is anthropic)
"""

import base64
import json
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _media_type(path: Path) -> str:
    """Return MIME type for image based on extension."""
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext in (".tif", ".tiff"):
        return "image/tiff"
    if ext in (".gif", ".webp"):
        return f"image/{ext[1:]}"
    return "image/png"


def _load_image_for_api(image_path: Path) -> tuple[bytes, str]:
    """Load image and return (base64_data, mime_type). Converts TIFF to PNG for API compatibility.
    Normalizes 16-bit microscopy images so they don't appear black."""
    ext = image_path.suffix.lower()
    if ext in (".tif", ".tiff"):
        import io
        import numpy as np
        from PIL import Image

        # Load with tifffile for proper 16-bit handling
        try:
            import tifffile

            arr = tifffile.imread(str(image_path))
        except Exception:
            arr = np.array(Image.open(image_path))
        arr = np.atleast_2d(arr)
        if arr.ndim > 2:
            arr = arr.squeeze()
            if arr.ndim > 2:
                arr = arr[0]
        arr = arr.astype(np.float64)
        # Rescale to 0-255 using percentiles (avoids black images from narrow range)
        p2, p98 = np.percentile(arr, (2, 98))
        if p98 > p2:
            arr = np.clip((arr - p2) / (p98 - p2) * 255, 0, 255)
        else:
            arr = np.clip(arr - arr.min(), 0, 255) if arr.max() > arr.min() else arr
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/png"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, _media_type(image_path)


PHENOTYPE_SCHEMA = {
    "type": "object",
    "properties": {
        "nuclei_count": {
            "type": "integer",
            "description": "Estimated number of nuclei. Use 0 if not a microscopy image or not applicable.",
        },
        "morphology": {
            "type": "string",
            "description": "Description of nuclei morphology (shape, size, distribution).",
        },
        "phenotype": {
            "type": "string",
            "description": "Phenotypic features observed (chromatin, staining, abnormalities).",
        },
    },
    "required": ["nuclei_count", "morphology", "phenotype"],
    "additionalProperties": False,
}


def get_vlm_description_openai(
    image_path: Path, prompt: str, model: str, api_key: str, structured: bool = False
) -> str | dict:
    """Get phenotype description from OpenAI vision model.
    If structured=True, returns dict with nuclei_count, morphology, phenotype.
    Otherwise returns free-form string."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    image_data, media = _load_image_for_api(image_path)

    kwargs = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{image_data}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }
    if structured:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "phenotype_analysis",
                "strict": True,
                "schema": PHENOTYPE_SCHEMA,
            },
        }
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content.strip()
    if structured:
        return json.loads(content)
    return content


def get_joint_vlm_description_openai(
    microscopy_path: Path,
    hic_path: Path,
    prompt: str,
    model: str,
    api_key: str,
) -> str:
    """Send both Hi-C map and microscopy image to OpenAI for joint generative phenotyping.
    Returns free-form text (no structured schema)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    mic_data, mic_media = _load_image_for_api(microscopy_path)
    hic_data, hic_media = _load_image_for_api(hic_path)

    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{hic_media};base64,{hic_data}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mic_media};base64,{mic_data}"},
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def get_vlm_description_anthropic(
    image_path: Path, prompt: str, model: str, api_key: str
) -> str:
    """Get phenotype description from Anthropic vision model."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    image_data, media = _load_image_for_api(image_path)

    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return response.content[0].text.strip()


def get_embedding(text: str, api_key: str) -> list[float]:
    """Embed text using OpenAI text-embedding-3-small."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def analyze_images(
    image_paths: list[Path],
    api_key: str,
    config: dict,
) -> pd.DataFrame | None:
    """Analyze a list of images with VLM. Returns DataFrame with image, description, embedding.
    If structured_output=True (OpenAI only), adds nuclei_count, morphology, phenotype columns."""
    if not image_paths or not api_key:
        return None
    vlm_cfg = config.get("vlm", {})
    backend = vlm_cfg.get("backend", "openai")
    model = vlm_cfg.get("model", "gpt-4o")
    structured = vlm_cfg.get("structured_output", False) and backend == "openai"
    prompt = vlm_cfg.get(
        "prompt",
        "Describe the nuclei morphology, chromatin distribution, and any notable "
        "phenotypic features in this fluorescence microscopy image. Be concise.",
    )
    rows = []
    last_error = None
    for path in image_paths:
        try:
            if backend == "openai":
                result = get_vlm_description_openai(
                    path, prompt, model, api_key, structured=structured
                )
            elif backend == "anthropic":
                result = get_vlm_description_anthropic(path, prompt, model, api_key)
            else:
                return None
            if structured and isinstance(result, dict):
                description = f"{result.get('morphology', '')} {result.get('phenotype', '')}".strip()
                row = {
                    "image": path.name,
                    "description": description,
                    "embedding": json.dumps(get_embedding(description, api_key)),
                    "nuclei_count": result.get("nuclei_count", 0),
                    "morphology": result.get("morphology", ""),
                    "phenotype": result.get("phenotype", ""),
                }
            else:
                description = str(result)
                row = {
                    "image": path.name,
                    "description": description,
                    "embedding": json.dumps(get_embedding(description, api_key)),
                }
            rows.append(row)
        except Exception as e:
            last_error = e
            print(f"  Error {path.name}: {e}")
    if not rows:
        if last_error:
            raise RuntimeError(f"Analysis failed: {last_error}") from last_error
        return None
    return pd.DataFrame(rows)


def run_pipeline(
    config: dict,
    api_key_override: str | None = None,
    overlay_dir_override: Path | None = None,
) -> Path | None:
    """Run VLM analysis on microscopy overlays.

    Args:
        config: Pipeline configuration dict.
        api_key_override: Optional API key (e.g. from Streamlit). When provided,
            used for OpenAI VLM and embeddings. For Anthropic backend, set
            api_key_env to ANTHROPIC_API_KEY and provide OpenAI key via env for embeddings.
        overlay_dir_override: Optional overlay directory (e.g. deploy_data when output missing).
    """
    output_dir = Path(config["paths"]["output_dir"]) / "microscopy"
    overlay_dir = overlay_dir_override or (output_dir / "overlays")

    if not overlay_dir.exists():
        print(f"Overlay directory not found: {overlay_dir}")
        print(
            "Run the microscopy pipeline first: python scripts/03_microscopy_pipeline.py"
        )
        return None

    overlay_paths = sorted(overlay_dir.glob("*.png"))
    if not overlay_paths:
        print(f"No overlay images in {overlay_dir}")
        return None

    vlm_cfg = config.get("vlm", {})
    backend = vlm_cfg.get("backend", "openai")
    model = vlm_cfg.get("model", "gpt-4o")
    api_key_env = vlm_cfg.get("api_key_env", "OPENAI_API_KEY")
    max_images = vlm_cfg.get("max_images", 5)
    structured = vlm_cfg.get("structured_output", False) and backend == "openai"
    prompt = vlm_cfg.get(
        "prompt",
        "Describe the nuclei morphology, chromatin distribution, and any notable "
        "phenotypic features in this fluorescence microscopy image. Be concise.",
    )

    api_key = api_key_override or os.environ.get(api_key_env)
    if not api_key:
        print(
            f"API key not found. Set {api_key_env} environment variable or pass api_key_override."
        )
        return None

    # Embeddings always use OpenAI; use override if provided, else env
    openai_key = api_key_override or os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print(
            "OPENAI_API_KEY required for embeddings (used regardless of VLM backend)."
        )
        return None

    if max_images is not None:
        overlay_paths = overlay_paths[:max_images]

    print(f"Processing {len(overlay_paths)} images with {backend} ({model})...")
    rows = []

    for path in overlay_paths:
        try:
            if backend == "openai":
                result = get_vlm_description_openai(
                    path, prompt, model, api_key, structured=structured
                )
            elif backend == "anthropic":
                result = get_vlm_description_anthropic(path, prompt, model, api_key)
            else:
                print(f"Unknown VLM backend: {backend}")
                return None
            if structured and isinstance(result, dict):
                description = f"{result.get('morphology', '')} {result.get('phenotype', '')}".strip()
                embedding = get_embedding(description, openai_key)
                rows.append(
                    {
                        "image": path.name,
                        "description": description,
                        "embedding": json.dumps(embedding),
                        "nuclei_count": result.get("nuclei_count", 0),
                        "morphology": result.get("morphology", ""),
                        "phenotype": result.get("phenotype", ""),
                    }
                )
            else:
                description = str(result)
                rows.append(
                    {
                        "image": path.name,
                        "description": description,
                        "embedding": json.dumps(get_embedding(description, openai_key)),
                    }
                )
            print(f"  Done: {path.name}")

        except Exception as e:
            print(f"  Error {path.name}: {e}")

    if not rows:
        print("No results generated.")
        return None

    df = pd.DataFrame(rows)
    out_path = output_dir / "vlm_output.csv"
    df.to_csv(out_path, index=False)
    print(f"Output saved to {out_path}")
    return output_dir


def main():
    config = load_config()
    os.chdir(ROOT)
    run_pipeline(config)


if __name__ == "__main__":
    main()
