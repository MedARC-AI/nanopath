import copy
import json
import sys
from pathlib import Path

import yaml


def deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def render_manifest(manifest_path: Path):
    manifest = yaml.safe_load(manifest_path.read_text())
    base_config_path = Path(manifest["base_config"])
    base_cfg = yaml.safe_load(base_config_path.read_text())
    generated_dir = Path(manifest["generated_config_dir"])
    output_root = Path(manifest["run_output_root"])
    generated_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_name = str(manifest["name"])
    base_name = str(base_cfg["project"]["name"])
    base_recipe = str(base_cfg["project"]["recipe_id"])
    index = []
    for variant in manifest["variants"]:
        slug = str(variant["slug"])
        cfg = deep_merge(base_cfg, variant["overrides"])
        cfg["project"]["name"] = f"{base_name}-{sweep_name}-{slug}"
        cfg["project"]["recipe_id"] = f"{base_recipe}-{sweep_name}-{slug}"
        cfg["project"]["output_dir"] = str(output_root / slug)
        config_path = generated_dir / f"{slug}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        index.append(
            {
                "slug": slug,
                "description": variant["description"],
                "config_path": str(config_path),
                "output_dir": cfg["project"]["output_dir"],
                "project_name": cfg["project"]["name"],
                "recipe_id": cfg["project"]["recipe_id"],
            }
        )
    index_path = generated_dir / "index.json"
    index_path.write_text(json.dumps({"manifest": str(manifest_path), "name": sweep_name, "variants": index}, indent=2))
    return index_path


def main():
    if len(sys.argv) != 2:
        raise ValueError("usage: python scripts/render_sweep.py <manifest>")
    index_path = render_manifest(Path(sys.argv[1]))
    print(index_path)


if __name__ == "__main__":
    main()
