# interpret/dashboard.py: builds the self-contained dashboard.html of a run from its results.pt, its plots and README.md.
# write_dashboard is the entry point. For RELAX and attention_map, the maps are layers over the input, each with an opacity slider.

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


def encode_image_base64(image_or_bytes, format="PNG"):
    if isinstance(image_or_bytes, (bytes, bytearray)):
        return base64.b64encode(image_or_bytes).decode()
    buffer = io.BytesIO()
    image_or_bytes.save(buffer, format=format)
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
.layer-row { display: grid; gap: 0.75rem; width: 100%; margin-bottom: 1.5rem; }
.panel { display: flex; flex-direction: column; gap: .3rem; min-width: 0; font-size: .85rem; }
.layers { width: 100%; background-size: 100% 100%; }
.layers img { display: block; width: 100%; image-rendering: pixelated; }
.scale { display: inline-block; width: 6rem; height: .7rem; border-radius: 3px; }
"""


# One panel per layer: the input (embedded once per row as a CSS background) with the layer over it at half opacity, and a slider that sets the layer's opacity.
# A layer is a (id, name, PNG, colour scale text) tuple; the colour scale is the CSS gradient of its colormap.
def layer_row_html(row_id, image_jpeg, layers, gradient):
    style = f"<style>.{row_id} {{ background-image: url(data:image/jpeg;base64,{image_jpeg}); }}</style>"
    panels = ""
    for layer_id, name, layer_png, scale_text in layers:
        overlay = f"<img id='{layer_id}' src='data:image/png;base64,{layer_png}' alt='{name}' style='opacity: 0.5'>"
        slider = f"<input type='range' min='0' max='100' value='50' oninput=\"document.getElementById('{layer_id}').style.opacity = this.value / 100\">"
        scale = f"<span><span class='scale' style='background: {gradient}'></span> {scale_text}</span>"
        panels += f"<div class='panel'><div class='layers {row_id}'>{overlay}</div><b>{name}</b>{slider}{scale}</div>"
    return f"{style}<div class='layer-row' style='grid-template-columns: repeat({len(layers)}, 1fr)'>{panels}</div>"


def colormap_gradient(name):
    stops = [colormaps[name](position) for position in (0.0, 0.25, 0.5, 0.75, 1.0)]
    return f"linear-gradient(to right, {', '.join(f'rgb({round(r * 255)}, {round(g * 255)}, {round(b * 255)})' for r, g, b, _ in stops)})"


# The RELAX maps for the dashboard, built from its results (not from its static figures): one row per image with three layers, the importance, the uncertainty and the thresholded
# importance, each as a coolwarm heat map (cold = low, warm = high) with its own colour scale and value range; the uncertain pixels of the thresholded map are transparent.
def relax_layers_html(relax_results, prefix="relax"):
    colormap = colormaps["coolwarm"]
    page_part = ""
    for image_index in relax_results["image_indices"]:
        maps = relax_results["maps"][image_index]
        image_channels_last = maps["image"].permute(1, 2, 0).numpy()  # (H, W, 3) in [0, 1]
        image_jpeg = encode_image_base64(Image.fromarray((image_channels_last * 255).round().astype(np.uint8)), format="JPEG")
        layers = []
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
            layer_png = encode_image_base64(Image.fromarray((rgba * 255).round().astype(np.uint8)))
            layers.append((f"{prefix}_{image_index}_{map_name}", f"image {image_index}: {map_name.replace('_', ' ')}", layer_png, f"{low:.3g} to {high:.3g}"))
        page_part += layer_row_html(f"{prefix}_{image_index}", image_jpeg, layers, colormap_gradient("coolwarm"))
    return page_part


# The attention maps for the dashboard, one row per tile with one layer per head of the last block: the CLS-to-patch attention on the patch grid, which the browser
# stretches over the tile with pixelated rendering (each patch stays a square), with one colour scale shared by the heads of a tile.
def attention_layers_html(attention_results, prefix="attention"):
    colormap = colormaps["Blues"]
    page_part = ""
    for tile_index in attention_results["tile_indices"]:
        tile = attention_results["tiles"][tile_index]
        image = tile["rgb"].permute(1, 2, 0).numpy()  # (H, W, 3) in [0, 1]
        image_jpeg = encode_image_base64(Image.fromarray((image * 255).round().astype(np.uint8)), format="JPEG")
        heads, num_patches = tile["attn"].shape
        grid = tile["attn"].reshape(heads, int(num_patches ** 0.5), -1).numpy()  # (heads, g, g)
        low, high = grid.min(), grid.max()
        normalized = (grid - low) / (high - low)  # [0, 1] shared by the heads of this tile
        layers = []
        for head, values in enumerate(normalized):
            layer_png = encode_image_base64(Image.fromarray((colormap(values) * 255).round().astype(np.uint8)))
            layers.append((f"{prefix}_{tile_index}_head_{head}", f"tile {tile_index}: head {head}", layer_png, f"{low:.3g} to {high:.3g}"))
        page_part += layer_row_html(f"{prefix}_{tile_index}", image_jpeg, layers, colormap_gradient("Blues"))
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
    if method_name == "attention_map" and results and "attention_map" in results:
        headline_plots = attention_layers_html(results["attention_map"], prefix=prefix)
    return headline_plots, debug_plots


# Writes results_dir/dashboard.html, to open with a double click: the settings of the run, then one section per method with its plots (embedded in the file,
# so it can be moved or sent around) and the questions and papers of that method's subsection of README.md as toggles (its notes on the paper(s) are private and left out of the dashboard).
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
            <div class='two-col-header'><div>Untrained Baseline</div><div>Trained Model</div></div>
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
            for part in re.split(r"(?m)^(?=Papers|Questions?|Notes on the paper)", method_text):
                if part.strip() and not part.strip().startswith("Notes on the paper"):
                    title, _, part_body = part.partition(":")
                    body += f"<details><summary>{title}</summary>{markdown.markdown(part_body, extensions=['tables'])}</details>"
        body += "</section>"

    page = f"<!doctype html><html><head><meta charset='utf-8'><title>Interpretability run</title><style>{STYLE}</style></head><body>{body}</body></html>"
    (results_dir / "dashboard.html").write_text(page)