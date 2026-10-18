#!/usr/bin/env bash
# =============================================================================
# ManiSkill3 演示 → LeRobotDataset 一键转换
#
# 为什么需要这个脚本：
#   本机 ManiSkill3 装在 rpd 环境（py3.11），LeRobot 装在 lerobot 环境（py3.12），
#   两个环境无法合并。本脚本负责用各自的解释器依次跑两段。
#
# 用法：
#   ./convert.sh --traj ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
#                --name pickcube --count 100
#
#   全部参数：
#     --traj PATH        演示轨迹 .h5（必需）
#     --name NAME        数据集短名，用于生成路径与 repo_id（必需）
#     --count N          处理多少个 episode，默认全部
#     --fps N            帧率，默认 20
#     --image-size N     图像分辨率，默认 256
#     --intermediate DIR 中间产物目录，默认 ~/datasets/intermediate/<name>
#     --root DIR         数据集目录，默认 ~/datasets/lerobot/<name>
#     --repo-id ID       默认 local/<name>
#     --dry-run          只回放 2 个 episode 验证链路，不落盘
#     --verify           打包后自检
#     --keep-intermediate 保留中间产物（默认转换成功后删除以省空间）
# =============================================================================
set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RPD_PY="${RPD_PY:-$HOME/miniconda3/envs/rpd/bin/python}"
LR_PY="${LR_PY:-$HOME/miniconda3/envs/lerobot/bin/python}"

TRAJ=""; NAME=""; COUNT="-1"; FPS="20"; IMAGE_SIZE="256"
INTERMEDIATE=""; ROOT=""; REPO_ID=""
DRY_RUN=""; VERIFY=""; KEEP_INTERMEDIATE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --traj) TRAJ="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --image-size) IMAGE_SIZE="$2"; shift 2 ;;
    --intermediate) INTERMEDIATE="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --repo-id) REPO_ID="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --verify) VERIFY="1"; shift ;;
    --keep-intermediate) KEEP_INTERMEDIATE="1"; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$TRAJ" ]] || { echo "缺少 --traj" >&2; exit 2; }
[[ -n "$NAME" ]] || { echo "缺少 --name" >&2; exit 2; }

INTERMEDIATE="${INTERMEDIATE:-$HOME/datasets/intermediate/$NAME}"
ROOT="${ROOT:-$HOME/datasets/lerobot/$NAME}"
REPO_ID="${REPO_ID:-local/$NAME}"

# --- 前置检查 ---------------------------------------------------------------
for py in "$RPD_PY" "$LR_PY"; do
  [[ -x "$py" ]] || { echo "找不到解释器: $py" >&2; echo "可用 RPD_PY / LR_PY 环境变量覆盖" >&2; exit 2; }
done

echo "=============================================================="
echo " Stage 1  回放 + 渲染   ($RPD_PY)"
echo "=============================================================="
ARGS=(--traj "$TRAJ" --out "$INTERMEDIATE" --count "$COUNT"
      --fps "$FPS" --image-size "$IMAGE_SIZE")
[[ -n "$DRY_RUN" ]] && ARGS+=(--dry-run)
"$RPD_PY" "$TOOL_DIR/stage1_replay_render.py" "${ARGS[@]}"

if [[ -n "$DRY_RUN" ]]; then
  echo
  echo "[dry-run] Stage 1 通过，跳过打包。去掉 --dry-run 执行完整转换。"
  exit 0
fi

echo
echo "=============================================================="
echo " Stage 2  打包 LeRobotDataset   ($LR_PY)"
echo "=============================================================="
ARGS=(--intermediate "$INTERMEDIATE" --root "$ROOT" --repo-id "$REPO_ID")
[[ -n "$VERIFY" ]] && ARGS+=(--verify)
"$LR_PY" "$TOOL_DIR/stage2_pack_lerobot.py" "${ARGS[@]}"

if [[ -z "$KEEP_INTERMEDIATE" ]]; then
  echo
  echo "[清理] 删除中间产物 $INTERMEDIATE（--keep-intermediate 可保留）"
  rm -rf "$INTERMEDIATE"
fi

echo
echo "=============================================================="
echo " 完成"
echo "=============================================================="
echo " 数据集路径: $ROOT"
echo " repo_id   : $REPO_ID"
echo
echo " 训练示例："
echo "   conda activate lerobot"
echo "   lerobot-train \\"
echo "     --policy.path=lerobot/smolvla_base \\"
echo "     --dataset.repo_id=$REPO_ID \\"
echo "     --dataset.root=$ROOT \\"
echo "     --batch_size=64 --steps=20000 \\"
echo "     --output_dir=outputs/train/smolvla_$NAME \\"
echo "     --job_name=smolvla_$NAME --policy.device=cuda"
