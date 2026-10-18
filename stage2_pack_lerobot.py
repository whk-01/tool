#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 —— 把 stage1 的中间产物打包成 LeRobotDataset（可直接喂给 lerobot-train）。

运行环境：lerobot（需要 lerobot 0.5.x，无需 mani_skill）

为什么用 LeRobotDataset.create 而不是手写 parquet/mp4？
    元数据（info.json / episodes / tasks / stats）、视频编码、分块与统计量都由
    LeRobot 自己维护，手写极易出错且随版本变化。这里只负责"喂帧"。

用法：
    python stage2_pack_lerobot.py \
        --intermediate ~/datasets/intermediate/pickcube \
        --root         ~/datasets/lerobot/pickcube \
        --repo-id      local/pickcube

    # 打包后自检：重新加载并核对帧数、维度、图像解码
    python stage2_pack_lerobot.py ... --verify
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="中间产物 → LeRobotDataset")
    ap.add_argument("--intermediate", required=True, help="stage1 输出目录（含 manifest.json）")
    ap.add_argument("--root", required=True, help="LeRobotDataset 落盘目录")
    ap.add_argument("--repo-id", required=True, help="数据集标识，形如 user/name")
    ap.add_argument("--camera-key", default="observation.images.human",
                    help="图像特征键名，默认 observation.images.human")
    ap.add_argument("--robot-type", default="panda", help="写入 meta 的机器人类型")
    ap.add_argument("--fps", type=int, default=None, help="覆盖 manifest 中的帧率")
    ap.add_argument("--no-videos", action="store_true",
                    help="用逐帧 PNG 而不是 MP4（体积大，仅调试用）")
    ap.add_argument("--state-mode", default="keep", choices=["keep", "drop"],
                    help="drop 表示不写 observation.state")
    ap.add_argument("--limit", type=int, default=-1, help="只打包前 N 个 episode")
    ap.add_argument("--verify", action="store_true", help="打包后重新加载自检")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    inter = Path(args.intermediate).expanduser()
    manifest_path = inter / "manifest.json"
    if not manifest_path.exists():
        print(f"[错误] 找不到 manifest.json: {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fps = args.fps or manifest["fps"]
    state_dim = int(manifest["state_dim"])
    action_dim = int(manifest["action_dim"])
    h = w = int(manifest["image_size"])
    task_text = manifest["task_text"]

    use_state = args.state_mode == "keep" and state_dim > 0
    if args.state_mode == "keep" and state_dim == 0:
        print("[警告] 中间产物没有本体状态，自动跳过 observation.state", file=sys.stderr)

    print(f"[信息] 环境 {manifest['env_id']}｜{manifest['total_episodes']} 个 episode"
          f"｜动作 {action_dim} 维｜状态 {'有' if use_state else '无'}｜{h}x{w}@{fps}fps")

    dtype_img = "image" if args.no_videos else "video"
    features = {
        args.camera_key: {
            "dtype": dtype_img,
            "shape": (3, h, w),
            "names": ["channels", "height", "width"],
        },
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
    }
    if use_state:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": None,
        }

    root = Path(args.root).expanduser()
    if root.exists() and any(root.iterdir()):
        print(f"[错误] 输出目录非空，请先清空或换个路径: {root}", file=sys.stderr)
        return 2
    root.mkdir(parents=True, exist_ok=True)

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type=args.robot_type,
        use_videos=not args.no_videos,
    )

    episodes = manifest["episodes"]
    if args.limit > 0:
        episodes = episodes[: args.limit]

    total_frames = 0
    for n, rec in enumerate(episodes, 1):
        ep_dir = inter / rec["dir"]
        frames = np.load(ep_dir / "frames.npy")
        actions = np.load(ep_dir / "actions.npy")
        states = np.load(ep_dir / "states.npy") if use_state else None

        if frames.shape[0] != actions.shape[0]:
            print(f"[错误] {rec['dir']} 帧数 {frames.shape[0]} 与动作数 {actions.shape[0]} 不一致",
                  file=sys.stderr)
            return 1
        if actions.shape[1] != action_dim:
            print(f"[错误] {rec['dir']} 动作维度 {actions.shape[1]} 与 manifest 的 {action_dim} 不一致",
                  file=sys.stderr)
            return 1

        for t in range(frames.shape[0]):
            frame = {
                args.camera_key: frames[t],  # HWC uint8
                "action": actions[t].astype(np.float32),
                "task": task_text,
            }
            if use_state:
                frame["observation.state"] = states[t].astype(np.float32)
            ds.add_frame(frame)

        ds.save_episode()
        total_frames += frames.shape[0]
        print(f"  [{n}/{len(episodes)}] {rec['dir']}: {frames.shape[0]} 帧已写入")

    ds.finalize()
    print(f"\n[完成] {len(episodes)} 个 episode，共 {total_frames} 帧 → {root}")
    print(f"        repo_id: {args.repo_id}")

    if args.verify:
        print("\n[自检] 重新加载数据集 ...")
        try:
            check = LeRobotDataset(args.repo_id, root=root)
            print(f"  episodes = {check.num_episodes}")
            print(f"  frames   = {check.num_frames}")
            sample = check[0]
            for k, v in sample.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            img = sample[args.camera_key]
            arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
            print(f"  图像解码正常，像素范围 [{arr.min()}, {arr.max()}]")
            print("  [通过]")
        except Exception as e:
            print(f"  [失败] 自检异常: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
