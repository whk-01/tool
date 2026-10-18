#!/usr/bin/env bash
# =============================================================================
# 诊断 SAPIEN 渲染可用性（WSL / Linux 通用）
#
# 它会做三件事：
#   1. 列出系统里的 Vulkan ICD 驱动
#   2. 检查 NVIDIA 图形库是否存在
#   3. 逐个 ICD 真实调用一次 SAPIEN 渲染器，看哪个能用
#
# 用法：
#   bash diagnose_vulkan.sh
#   RPD_PY=/path/to/python bash diagnose_vulkan.sh
# =============================================================================
set -uo pipefail

RPD_PY="${RPD_PY:-$HOME/miniconda3/envs/rpd/bin/python}"

echo "=============================================================="
echo " 1. 解释器"
echo "=============================================================="
if [[ ! -x "$RPD_PY" ]]; then
  echo "  ✗ 找不到 $RPD_PY"
  echo "    用 RPD_PY=/path/to/python bash diagnose_vulkan.sh 指定"
  exit 2
fi
echo "  $RPD_PY"
"$RPD_PY" -c "import sys; print('  Python', sys.version.split()[0])"

echo
echo "=============================================================="
echo " 2. Vulkan ICD 驱动清单"
echo "=============================================================="
ICD_DIR="/usr/share/vulkan/icd.d"
if [[ -d "$ICD_DIR" ]]; then
  shopt -s nullglob
  ICDS=("$ICD_DIR"/*.json)
  if [[ ${#ICDS[@]} -eq 0 ]]; then
    echo "  ✗ 目录为空，没有安装任何 Vulkan 驱动"
  else
    for f in "${ICDS[@]}"; do
      base="$(basename "$f")"
      case "$base" in
        nvidia_icd*)    tag="← NVIDIA 官方（硬件，最优）" ;;
        dzn_icd*)       tag="← Dozen（Vulkan→D3D12，WSL 推荐）" ;;
        lvp_icd*)       tag="← lavapipe（纯 CPU 软件渲染，很慢）" ;;
        *)              tag="" ;;
      esac
      printf "  %-28s %s\n" "$base" "$tag"
    done
  fi
else
  echo "  ✗ $ICD_DIR 不存在"
fi

echo
echo "=============================================================="
echo " 3. NVIDIA 图形库"
echo "=============================================================="
for f in libcuda.so.1 libGLX_nvidia.so.0 libnvidia-vulkan-producer.so; do
  found=""
  for d in /usr/lib/wsl/lib /usr/lib/x86_64-linux-gnu; do
    [[ -e "$d/$f" ]] && found="$d/$f" && break
  done
  if [[ -n "$found" ]]; then
    printf "  ✓ %-34s %s\n" "$f" "$found"
  else
    printf "  ✗ %-34s 缺失\n" "$f"
  fi
done

echo
echo "=============================================================="
echo " 4. 逐个 ICD 实测 SAPIEN 渲染器"
echo "=============================================================="

PROBE='
import sys
cls = None
try:
    import sapien.render as sr
    cls = getattr(sr, "SapienRenderer", None)
except Exception:
    pass
if cls is None:
    try:
        import sapien
        cls = getattr(sapien, "SapienRenderer", None)
    except Exception:
        pass
if cls is None:
    print("SKIP: 找不到 SapienRenderer 类")
    sys.exit(3)
try:
    r = cls()
    name = ""
    for attr in ("get_device_name", "device_name"):
        fn = getattr(r, attr, None)
        if callable(fn):
            try:
                name = str(fn()); break
            except Exception:
                pass
    print("OK" + ((" | " + name) if name else ""))
except Exception as e:
    print("FAIL: " + type(e).__name__ + ": " + str(e)[:200])
    sys.exit(1)
'

if [[ ${#ICDS[@]} -eq 0 ]]; then
  echo "  跳过（没有 ICD）"
else
  ok_count=0
  for f in "${ICDS[@]}"; do
    base="$(basename "$f")"
    out="$(VK_ICD_FILENAMES="$f" "$RPD_PY" -c "$PROBE" 2>&1 | tail -1)"
    if [[ "$out" == OK* ]]; then
      printf "  ✓ %-28s %s\n" "$base" "$out"
      ok_count=$((ok_count + 1))
    else
      printf "  ✗ %-28s %s\n" "$base" "$out"
    fi
  done
fi

echo
echo "=============================================================="
echo " 结论"
echo "=============================================================="
if [[ ${ok_count:-0} -gt 0 ]]; then
  echo "  至少有一个 ICD 能让 SAPIEN 建起渲染器。"
  echo "  用对应的环境变量跑转换："
  echo
  echo "    export VK_ICD_FILENAMES=<上面打 ✓ 的那个 json>"
  echo "    cd ~/lerobot/tool && bash convert.sh --traj ... --name ... --count 2 --dry-run"
else
  echo "  没有任何 ICD 能让 SAPIEN 建起渲染器。"
  echo
  echo "  这意味着本机（当前这个 WSL 发行版）无法做 SAPIEN 渲染，"
  echo "  stage1 必须换到别处跑。可选："
  echo
  echo "    A. Windows 原生：建 Python 环境装 mani_skill，在那里跑 stage1"
  echo "       （SAPIEN 有 Windows 轮子，图形驱动齐全）"
  echo "    B. 原生 Linux + 独显：装完整 NVIDIA 驱动（不要 -server 版）"
  echo "    C. 已有一台能跑的 Linux 机器：把 stage1 搬过去，产物拷回来跑 stage2"
  echo
  echo "  注意：stage2（打包）不需要渲染，留在本机跑即可。"
fi
echo
