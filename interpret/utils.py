# interpret/utils.py: helpers of the interpretability rig.
# tissue_masks finds the tissue pixels of a batch of tiles. write_dashboard builds the self-contained dashboard.html of a run: its settings, and per method the plots and the README parts as toggles;
# for RELAX, relax_layers_html shows the maps as layers over the image, each with an opacity slider.

import base64
import html
import io
import json
import re
from pathlib import Path

import markdown
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from prepare import tissue_mask

# Maximum dimension (width/height) for images embedded in the dashboard HTML.
# Downscaling keeps dashboard.html lightweight (a few MBs instead of tens of MBs) while maintaining visual clarity.
DASHBOARD_MAX_IMAGE_DIM = 1500


def encode_image_base64(image_or_bytes):
    image = Image.open(io.BytesIO(image_or_bytes)) if isinstance(image_or_bytes, (bytes, bytearray)) else image_or_bytes
    if DASHBOARD_MAX_IMAGE_DIM and max(image.size) > DASHBOARD_MAX_IMAGE_DIM:
        image = image.copy()
        image.thumbnail((DASHBOARD_MAX_IMAGE_DIM, DASHBOARD_MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()

# Style of the dashboard: the browser's own light/dark colors, a readable column, plots side by side.
STYLE = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; max-width: 1400px; margin: 2rem auto; padding: 0 1rem; line-height: 1.55; }
h1 { margin-bottom: 1rem; }
h2 { margin: 0 0 1rem; padding-bottom: .5rem; border-bottom: 2px solid #8886; font-size: 1.5rem; }
table { border-collapse: collapse; }
th, td { border: 1px solid #8884; padding: .3rem .6rem; text-align: left; vertical-align: top; }
tr:nth-child(even) td { background: #8881; }
.card { background: #8881; border: 1px solid #8883; border-radius: 12px; padding: 1.2rem 1.5rem; margin: 2.5rem 0; box-shadow: 0 2px 8px #0002; }
details { border: 1px solid #8884; border-radius: 8px; padding: .4rem .8rem; margin: .6rem 0; overflow-x: auto; }
details[open] { background: #8881; }
summary { cursor: pointer; font-weight: 600; }
summary:hover { opacity: .7; }
.plots { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem; }
.plots img { max-width: 100%; height: auto; background: white; border-radius: 8px; box-shadow: 0 1px 4px #0003; }
.two-col-header { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 0.5rem; font-weight: 700; font-size: 1.1rem; text-align: center; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; align-items: start; }
.col { display: flex; flex-direction: column; gap: 1rem; min-width: 0; }
pre { overflow-x: auto; }
.relax-image { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; width: 100%; margin-bottom: 1.5rem; }
.panel { display: flex; flex-direction: column; gap: .3rem; min-width: 0; font-size: .85rem; }
.layers { position: relative; width: 100%; }
.layers img { display: block; width: 100%; max-width: none; background: none; box-shadow: none; border-radius: 0; }
.layers img + img { position: absolute; top: 0; left: 0; }
.scale { display: inline-block; width: 6rem; height: .7rem; border-radius: 3px; }
"""


# The tissue mask of a batch of tiles with the repo's rule (tissue_mask in prepare.py, which takes one tile at a time). The rule thresholds the saturation of RGB values in [0, 1], so a batch that is normalized, in 0-255
# or without the batch dimension would give a meaningless mask without any error: the preconditions are asserted here, every use of the tissue rule in the rig goes through this function.
def tissue_masks(images):  # (B, 3, H, W) floats in [0, 1] -> (B, H, W) booleans
    assert images.ndim == 4 and images.shape[1] == 3, f"expected tiles of shape (B, 3, H, W), got {tuple(images.shape)}"
    assert images.dtype.is_floating_point, f"expected float tiles in [0, 1], got {images.dtype}"
    assert images.min() >= 0.0 and images.max() <= 1.0, f"the tissue rule needs RGB in [0, 1] (not normalized, not 0-255), got the range [{images.min().item():.3g}, {images.max().item():.3g}]"
    return torch.stack([tissue_mask(tile) for tile in images])


# The RELAX maps for the dashboard, built from its results (not from its static figures): one row per image with four panels, the input, then the importance, the uncertainty and the
# thresholded importance, each as a coolwarm heat map (cold = low, warm = high) over its own copy of the input with its own opacity slider, colour scale and value range, so that all
# three can be shown at once and compared side by side; the uncertain pixels of the thresholded map are transparent.
def relax_layers_html(relax_results, prefix="relax"):
    colormap = colormaps["coolwarm"]
    gradient_stops = []
    for position in (0.0, 0.25, 0.5, 0.75, 1.0):
        rgba = colormap(position)
        gradient_stops.append(f"rgb({round(rgba[0] * 255)}, {round(rgba[1] * 255)}, {round(rgba[2] * 255)})")
    gradient = f"linear-gradient(to right, {', '.join(gradient_stops)})"

    page_part = ""
    for image_index in relax_results["image_indices"]:
        maps = relax_results["maps"][image_index]
        image_channels_last = maps["image"].permute(1, 2, 0).numpy()  # (H, W, 3) in [0, 1]
        layers = {"image": (image_channels_last * 255).round().astype(np.uint8)}  # {layer name: (H, W, 3 or 4) uint8}
        value_ranges = {}
        for map_name in ("importance", "uncertainty", "thresholded_importance"):
            values = maps[map_name].numpy()  # (H, W)
            is_shown = np.ones_like(values, dtype=bool)  # (H, W), the pixels that are coloured
            if map_name == "thresholded_importance":
                is_shown = values != 0  # the uncertain pixels (set to 0) are not coloured
            low = values[is_shown].min()
            high = values[is_shown].max()
            scaled = (values - low) / (high - low)  # [0, 1] over the shown range of this map
            rgba = colormap(scaled)  # (H, W, 4) in [0, 1]
            rgba[:, :, 3] = is_shown  # transparent where not shown
            layers[map_name] = (rgba * 255).round().astype(np.uint8)
            value_ranges[map_name] = (low, high)

        encoded = {}  # {layer name: the PNG of the layer as base64}
        for layer_name, pixels in layers.items():
            encoded[layer_name] = encode_image_base64(Image.fromarray(pixels))
        image_tag = f"<img src='data:image/png;base64,{encoded['image']}' alt='image'>"
        panels = f"<div class='panel'><div class='layers'>{image_tag}</div><b>image {image_index}</b></div>"
        for map_name in ("importance", "uncertainty", "thresholded_importance"):
            layer_id = f"{prefix}_{image_index}_{map_name}"
            low, high = value_ranges[map_name]
            overlay = f"<img id='{layer_id}' src='data:image/png;base64,{encoded[map_name]}' alt='{map_name}' style='opacity: 0.5'>"
            slider = f"<input type='range' min='0' max='100' value='50' oninput=\"document.getElementById('{layer_id}').style.opacity = this.value / 100\">"
            scale = f"<span><span class='scale' style='background: {gradient}'></span> {low:.3g} to {high:.3g}</span>"
            panels += f"<div class='panel'><div class='layers'>{image_tag}{overlay}</div><b>{map_name.replace('_', ' ')}</b>{slider}{scale}</div>"
        page_part += f"<div class='relax-image'>{panels}</div>"
    return page_part


def _render_method_plots(results_dir, method_name, results=None, prefix="plot"):
    headline_plots, debug_plots = "", ""
    method_dir = results_dir / method_name
    if method_dir.exists():
        for plot_path in sorted(method_dir.glob("*.png")):
            encoded = encode_image_base64(plot_path.read_bytes())
            img = f"<img src='data:image/png;base64,{encoded}' alt='{plot_path.stem}'>"
            if plot_path.name.startswith("debug_"):
                debug_plots += img
            else:
                headline_plots += img
    if method_name == "relax" and results and "relax" in results:
        headline_plots = relax_layers_html(results["relax"], prefix=prefix)
    return headline_plots, debug_plots


# Writes results_dir/dashboard.html, to open with a double click: the settings of the run, then one section per method with its plots (embedded in the file,
# so it can be moved or sent around) and the parts of that method's subsection of README.md (papers, questions, notes, ...) as toggles.
def write_dashboard(results_dir, cfg, baseline_dir=None, trained_dir=None):
    icfg = cfg["interpret"]
    readme = (Path(__file__).resolve().parent / "README.md").read_text()
    is_comparison = baseline_dir is not None and trained_dir is not None

    summary = {
        "checkpoint": icfg["checkpoint_path"], "weights": icfg["model_weights"], "dataset": icfg["dataset"],
        "seed": icfg["seed"],
    }
    if is_comparison:
        summary["baseline"] = icfg.get("baseline_checkpoint_path", "init.pt")
        baseline_res = torch.load(baseline_dir / "results.pt", weights_only=False)
        trained_res = torch.load(trained_dir / "results.pt", weights_only=False)
    else:
        trained_res = torch.load(results_dir / "results.pt", weights_only=False)

    summary_rows = "".join(f"<tr><th>{k}</th><td>{html.escape(str(v))}</td></tr>" for k, v in summary.items())
    body = f"<h1>Interpretability run</h1><table>{summary_rows}</table>"
    body += f"<details><summary>full config</summary><pre>{html.escape(json.dumps(cfg, indent=2))}</pre></details>"

    for method_name in icfg["methods"]:
        if is_comparison:
            b_head, b_debug = _render_method_plots(baseline_dir, method_name, baseline_res, prefix=f"b_{method_name}")
            t_head, t_debug = _render_method_plots(trained_dir, method_name, trained_res, prefix=f"t_{method_name}")

            plots_html = f"""
            <div class='two-col-header'><div>Untrained Baseline (Init)</div><div>Trained Model (Final)</div></div>
            <div class='two-col'>
                <div class='col'>{b_head}</div>
                <div class='col'>{t_head}</div>
            </div>
            """
            debug_html = ""
            if b_debug or t_debug:
                debug_html = f"""
                <details><summary>debug plots</summary>
                <div class='two-col'>
                    <div class='col'>{b_debug}</div>
                    <div class='col'>{t_debug}</div>
                </div></details>
                """
        else:
            head, debug = _render_method_plots(results_dir, method_name, trained_res, prefix=method_name)
            plots_html = f"<div class='plots'>{head}</div>"
            debug_html = f"<details><summary>debug plots</summary><div class='plots'>{debug}</div></details>" if debug else ""

        body += f"<section class='card'><h2>{method_name}</h2>{plots_html}{debug_html}"

        method_text_match = re.search(rf"^### {method_name}\n(.*?)(?=^##|\Z)", readme, re.S | re.M)
        if method_text_match:
            method_text = method_text_match.group(1)
            for part in re.split(r"(?m)^(?=Papers|Question)", method_text):
                if part.strip() and not part.strip().startswith("Notes on the paper"):
                    title, _, part_body = part.partition(":")
                    body += f"<details><summary>{title}</summary>{markdown.markdown(part_body, extensions=['tables'])}</details>"
        body += "</section>"

    page = f"<!doctype html><html><head><meta charset='utf-8'><title>Interpretability run</title><style>{STYLE}</style></head><body>{body}</body></html>"
    (results_dir / "dashboard.html").write_text(page)
