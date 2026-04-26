#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: bash download_TCGA.sh [download_root] [n_processes]

Downloads the open-access TCGA SVS slide set with the GDC Data Transfer Tool
and downloads NanoPath's sample_dataset_30.txt from Hugging Face.

Defaults:
  download_root: /data/TCGA
  n_processes:   8

For the gated Hugging Face file, authenticate with one of:
  HF_TOKEN=... bash download_TCGA.sh /data/TCGA 8
  HUGGING_FACE_HUB_TOKEN=... bash download_TCGA.sh /data/TCGA 8
  huggingface-cli login

The final sample list is written to:
  <download_root>/sample_dataset_30.txt

The script normalizes sample-list slide paths to:
  <download_root>/<slide_filename>.svs
and creates flat symlinks from <download_root>/<slide_filename>.svs to the
GDC client's <download_root>/files/<file_id>/<slide_filename>.svs layout.
EOF
}

if [[ $# -gt 2 ]]; then
  usage
  exit 2
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This downloader is intentionally hardcoded for Linux x86_64, the expected target for a multi-TB TCGA download." >&2
  exit 1
fi

for cmd in awk basename cat chmod curl date df find grep head ln md5sum mkdir mv rm sed sort unzip wc; do
  if ! command -v "$cmd" >/dev/null; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

DOWNLOAD_ROOT="${1:-/data/TCGA}"
N_PROCESSES="${2:-8}"

mkdir -p "$DOWNLOAD_ROOT"
DOWNLOAD_ROOT="$(cd "$DOWNLOAD_ROOT" && pwd -P)"

GDC_CLIENT_URL="https://gdc.cancer.gov/system/files/public/file/gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04.zip"
GDC_CLIENT_MD5="18591d74de07cdcd396dab71c52663da"
TCGA_SVS_FILTERS="%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22%3D%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.program.name%22%2C%22value%22%3A%22TCGA%22%7D%7D%2C%7B%22op%22%3A%22%3D%22%2C%22content%22%3A%7B%22field%22%3A%22data_type%22%2C%22value%22%3A%22Slide%20Image%22%7D%7D%2C%7B%22op%22%3A%22%3D%22%2C%22content%22%3A%7B%22field%22%3A%22data_format%22%2C%22value%22%3A%22SVS%22%7D%7D%2C%7B%22op%22%3A%22%3D%22%2C%22content%22%3A%7B%22field%22%3A%22access%22%2C%22value%22%3A%22open%22%7D%7D%2C%7B%22op%22%3A%22%3D%22%2C%22content%22%3A%7B%22field%22%3A%22state%22%2C%22value%22%3A%22released%22%7D%7D%5D%7D"
SAMPLE_LIST_URL="https://huggingface.co/SophontAI/OpenMidnight/resolve/main/sample_dataset_30.txt"

TOOLS_DIR="$DOWNLOAD_ROOT/_gdc_tools"
CLIENT_ZIP="$TOOLS_DIR/gdc-client.zip"
CLIENT_DIR="$TOOLS_DIR/gdc-client"
GDC_CLIENT="$CLIENT_DIR/gdc-client"
MANIFEST="$DOWNLOAD_ROOT/tcga_svs_manifest.txt"
FILES_DIR="$DOWNLOAD_ROOT/files"
SAMPLE_LIST="$DOWNLOAD_ROOT/sample_dataset_30.txt"

mkdir -p "$TOOLS_DIR" "$CLIENT_DIR" "$FILES_DIR"

hf_token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
hf_token_file="${HF_HOME:-${HOME:-}/.cache/huggingface}/token"
if [[ -z "$hf_token" && -n "$hf_token_file" && -f "$hf_token_file" ]]; then
  hf_token="$(sed -n '1p' "$hf_token_file")"
fi
HF_CURL_ARGS=()
if [[ -n "$hf_token" ]]; then
  HF_CURL_ARGS=(-H "Authorization: Bearer $hf_token")
fi

normalize_sample_list() {
  local src="$1"
  local dst="$2"
  local tmp="${dst}.$$.$(date +%s).tmp"

  if ! awk 'NR == 1 {exit !(NF >= 4 && $1 ~ /\.svs$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/)}' "$src"; then
    echo "sample_dataset_30.txt did not look like a NanoPath sample list. First lines:" >&2
    head "$src" >&2
    exit 1
  fi

  awk -v root="$DOWNLOAD_ROOT" '
    NF >= 4 {
      n = split($1, parts, "/")
      $1 = root "/" parts[n]
      print
      next
    }
  ' "$src" > "$tmp"
  mv "$tmp" "$dst"
}

download_sample_list() {
  local first_path
  if [[ -s "$SAMPLE_LIST" ]]; then
    first_path="$(awk 'NF {print $1; exit}' "$SAMPLE_LIST")"
    if [[ "$first_path" == "$DOWNLOAD_ROOT/"* ]]; then
      echo "[skip] sample list already present: $SAMPLE_LIST"
      return
    fi
    echo "[fix] normalizing existing sample list paths to $DOWNLOAD_ROOT"
    normalize_sample_list "$SAMPLE_LIST" "$SAMPLE_LIST"
    return
  fi

  local raw="${SAMPLE_LIST}.download"
  echo "[fetch] sample_dataset_30.txt -> $SAMPLE_LIST"
  if ! curl -fL -C - "${HF_CURL_ARGS[@]}" "$SAMPLE_LIST_URL" -o "$raw"; then
    cat >&2 <<EOF
Failed to download sample_dataset_30.txt from Hugging Face.

The OpenMidnight Hugging Face repo may be gated. Request access, then retry with
HF_TOKEN set or run huggingface-cli login before this script.
If this was a transient network failure, the partial download remains at:
  $raw

URL: $SAMPLE_LIST_URL
EOF
    exit 1
  fi
  normalize_sample_list "$raw" "$SAMPLE_LIST"
  rm -f "$raw"
  echo "[done] sample list rows: $(wc -l < "$SAMPLE_LIST")"
}

install_gdc_client() {
  if [[ -x "$GDC_CLIENT" ]]; then
    echo "[skip] GDC client already installed: $GDC_CLIENT"
    return
  fi

  echo "[fetch] installing GDC Data Transfer Tool into $CLIENT_DIR"
  curl -fL "$GDC_CLIENT_URL" -o "$CLIENT_ZIP"
  echo "$GDC_CLIENT_MD5  $CLIENT_ZIP" | md5sum -c -
  unzip -q -o "$CLIENT_ZIP" -d "$CLIENT_DIR"
  if [[ ! -f "$GDC_CLIENT" ]]; then
    local found
    found="$(find "$CLIENT_DIR" -type f -name gdc-client | head -n 1)"
    if [[ -z "$found" ]]; then
      echo "Could not find gdc-client after unpacking $CLIENT_ZIP" >&2
      exit 1
    fi
    mv "$found" "$GDC_CLIENT"
  fi
  chmod +x "$GDC_CLIENT"
}

write_gdc_manifest() {
  echo "[fetch] generating TCGA SVS manifest at $MANIFEST"
  curl -fL "https://api.gdc.cancer.gov/files?filters=${TCGA_SVS_FILTERS}&return_type=manifest" -o "$MANIFEST"

  if ! head -n 1 "$MANIFEST" | grep -q $'^id\tfilename\tmd5\tsize\tstate'; then
    echo "Manifest did not have the expected GDC manifest header. First lines:" >&2
    head "$MANIFEST" >&2
    exit 1
  fi
}

summarize_manifest_and_space() {
  local file_count total_bytes total_tb total_tib avail_kb needed_kb
  file_count="$(awk 'NR > 1 {n += 1} END {print n + 0}' "$MANIFEST")"
  total_bytes="$(awk 'NR > 1 {s += $4} END {printf "%.0f", s + 0}' "$MANIFEST")"
  total_tb="$(awk -v b="$total_bytes" 'BEGIN {printf "%.2f", b / 1000000000000}')"
  total_tib="$(awk -v b="$total_bytes" 'BEGIN {printf "%.2f", b / 1099511627776}')"
  avail_kb="$(df -Pk "$DOWNLOAD_ROOT" | awk 'NR == 2 {print $4}')"
  needed_kb="$(awk -v b="$total_bytes" 'BEGIN {printf "%.0f", (b * 1.05) / 1024}')"

  if (( file_count == 0 )); then
    echo "The GDC query returned zero files; not starting download." >&2
    exit 1
  fi

  if (( avail_kb < needed_kb )); then
    echo "Warning: filesystem free space appears lower than manifest size + 5%." >&2
    echo "Available: $(awk -v kb="$avail_kb" 'BEGIN {printf "%.2f TiB", kb / 1073741824}')" >&2
    echo "Needed:    $(awk -v kb="$needed_kb" 'BEGIN {printf "%.2f TiB", kb / 1073741824}')" >&2
  fi

  echo "Manifest files: $file_count"
  echo "Manifest size:  $total_tb TB / $total_tib TiB"
  echo "Download dir:   $FILES_DIR"
  echo "Connections:    $N_PROCESSES"
}

download_slides() {
  echo "[fetch] starting resumable GDC download. Re-run this command to resume after interruption."
  "$GDC_CLIENT" download \
    -m "$MANIFEST" \
    -d "$FILES_DIR" \
    -n "$N_PROCESSES" \
    --retry-amount 20 \
    --wait-time 60 \
    --no-related-files \
    --no-annotations
}

link_flat_svs_paths() {
  echo "[link] creating flat SVS symlinks in $DOWNLOAD_ROOT"
  find "$FILES_DIR" -type f -name '*.svs' -print0 | sort -z | while IFS= read -r -d '' slide; do
    local_name="$(basename "$slide")"
    target="$DOWNLOAD_ROOT/$local_name"
    rel="${slide#"$DOWNLOAD_ROOT"/}"
    if [[ -L "$target" && ! -e "$target" ]]; then
      rm -f "$target"
    fi
    if [[ ! -e "$target" ]]; then
      ln -s "$rel" "$target"
    fi
  done
  echo "[done] flat SVS paths available: $(find "$DOWNLOAD_ROOT" -maxdepth 1 \( -type f -o -type l \) -name '*.svs' | wc -l)"
}

download_sample_list
install_gdc_client
write_gdc_manifest
summarize_manifest_and_space
download_slides
link_flat_svs_paths

first_slide="$(awk 'NF {print $1; exit}' "$SAMPLE_LIST")"
if [[ ! -e "$first_slide" ]]; then
  cat >&2 <<EOF
Warning: the first sample-list slide is still missing:
  $first_slide

The GDC download may not have completed, or the GDC file set may differ from
the sample list. Re-run this script to resume and then check missing files.
EOF
fi

cat <<EOF
[done] TCGA setup complete.

Sample list:
  $SAMPLE_LIST

For NanoPath configs that do not use the shared /block/TCGA path, set:
  data.sample_list: $SAMPLE_LIST
EOF
