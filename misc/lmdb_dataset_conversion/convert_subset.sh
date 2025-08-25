#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 -i INPUT_LMDB -o OUTPUT_EXTXYZ [-l LIMIT] [-k EVERY] [--pad]

  -i  输入 LMDB 路径
  -o  输出 EXTXYZ 路径（如给 .xyz，会自动写为 .extxyz）
  -l  最多转换多少条（可选）
  -k  每隔 k 条取一条（可选，默认 1）
  --pad  将可变宽改为按 n_max 全局补零（可选）

示例：
  $0 -i archive/ts1x_hess_train_big.lmdb -o archive/ts1x_hess_train_big.sub200.extxyz -l 200 -k 50
EOF
}

INPUT=""
OUTPUT=""
LIMIT=""
EVERY=1
PAD_FLAG=""

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i)
      INPUT="$2"; shift 2;;
    -o)
      OUTPUT="$2"; shift 2;;
    -l)
      LIMIT="$2"; shift 2;;
    -k)
      EVERY="$2"; shift 2;;
    --pad)
      PAD_FLAG="--pad-hessian"; shift 1;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
  echo "-i 与 -o 为必填"; usage; exit 1
fi

# 执行转换（默认可变宽 + 嵌入 Hessian）
CMD=(python -m nequip.misc.lmdb_dataset_conversion.lmdb_to_xyz \
  --in "$INPUT" \
  --out "$OUTPUT" \
  --pos pos --charges charges --forces forces --energy energy \
  --hessian hessian --embed-hessian $PAD_FLAG \
  --every "$EVERY")

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

printf "Running: %s\n" "${CMD[*]}"
"${CMD[@]}"
