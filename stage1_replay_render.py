#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 —— 回放 ManiSkill3 演示轨迹，渲染 human camera，导出中间产物。

为什么要分两段？
    本机 ManiSkill3 装在 `rpd` 环境（Python 3.11），LeRobot 装在 `lerobot`
    环境（Python 3.12），两者无法合并。所以本脚本只负责"把轨迹变成图像+数值"，
    不依赖 lerobot；打包成 LeRobotDataset 由 stage2 在另一个环境完成。

运行环境：rpd（需要 mani_skill / sapien / gymnasium / h5py）

输出结构：
    <out>/
    ├── manifest.json                  # 全局元信息 + episode 清单
    └── ep_000000/
        ├── frames.npy                 # (T, H, W, 3) uint8
        ├── states.npy                 # (T, state_dim) float32
        └── actions.npy                # (T, action_dim) float32

用法：
    python stage1_replay_render.py \
        --traj ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
        --out  ~/datasets/intermediate/pickcube \
        --count 100

    # 先干跑一遍，确认能建环境、能取到图，不落盘
    python stage1_replay_render.py --traj ... --out ... --count 2 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 与 RPD 源码（src/rpd/ppo_rgb_rpd.py）保持逐字一致的语言指令。
# 数据集里的 task 文本必须和训练时喂给 VLA 的一致，否则指令嵌入对不上。
# ---------------------------------------------------------------------------
INSTRUCTIONS = {
    "LiftPegUpright-v1": "lift the peg upright",
    "PegInsertionSide-v1": "insert the peg from the side",
    "PickCube-v1": "pick up the cube",
    "PlugCharger-v1": "plug the charger in",
    "PokeCube-v1": "push the cube by using the blue tool",
    "PullCube-v1": "pull the cube towards the robot base",
    "PullCubeTool-v1": "pull the cube by using the red tool",
    "PushCube-v1": "push the cube away from the robot base",
    "PushT-v1": "align the T shape",
    "RollBall-v1": "push the ball",
    "StackCube-v1": "stack the red cube on the green cube",
}


def to_hwc_uint8(x) -> np.ndarray:
    """把 env.render() 的各种返回形态统一成 (H, W, 3) uint8。

    可能遇到：torch.Tensor / np.ndarray / (N,H,W,3) / (N,3,H,W) / (3,H,W) / float[0,1]
    """
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass

    x = np.asarray(x)

    if x.ndim == 4:  # 多环境 → 取第一个
        x = x[0]
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:  # CHW → HWC
        x = x.transpose(1, 2, 0)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"无法识别的图像形状: {x.shape}")

    if x.dtype != np.uint8:
        # float 图通常是 [0,1]，整数图保持原值
        if np.issubdtype(x.dtype, np.floating) and x.max() <= 1.0 + 1e-6:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def get_proprio(env) -> np.ndarray:
    """取本体感受状态（关节位置）。取不到就返回空向量，由 stage2 决定是否丢弃。"""
    try:
        qpos = env.unwrapped.agent.robot.get_qpos()
        try:
            import torch

            if isinstance(qpos, torch.Tensor):
                qpos = qpos.detach().cpu().numpy()
        except ImportError:
            pass
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        return qpos
    except Exception:
        return np.zeros((0,), dtype=np.float32)


def build_env(env_id: str, base_kwargs: dict, image_size: int, sim_backend: str):
    """按 RPD 的配置建环境：rgb 观测 + pd_ee_delta_pose + 256 的 human camera。"""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  注册环境

    kwargs = dict(base_kwargs)
    kwargs.update(
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend=sim_backend,
        human_render_camera_configs={"width": image_size, "height": image_size},
    )
    # 回放不需要奖励，但某些环境要求 reward_mode 合法值
    kwargs.setdefault("reward_mode", "normalized_dense")
    return gym.make(env_id, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description="ManiSkill3 轨迹 → 图像+数值中间产物")
    ap.add_argument("--traj", required=True, help="演示轨迹 .h5 路径（同名 .json 必须存在）")
    ap.add_argument("--out", required=True, help="中间产物输出目录")
    ap.add_argument("--count", type=int, default=-1, help="最多处理多少个 episode，-1 表示全部")
    ap.add_argument("--image-size", type=int, default=256, help="human camera 分辨率（RPD 用 256）")
    ap.add_argument("--fps", type=int, default=20, help="写入 manifest 的帧率，需与演示一致")
    ap.add_argument("--sim-backend", default="cpu", choices=["cpu", "gpu"], help="单环境回放建议 cpu")
    ap.add_argument("--max-steps", type=int, default=0, help="每回合最多保留多少步，0 表示不截断")
    ap.add_argument("--task-text", default=None, help="覆盖默认语言指令")
    ap.add_argument("--dry-run", action="store_true", help="只跑前 2 个 episode 且不落盘")
    args = ap.parse_args()

    traj_path = Path(args.traj).expanduser()
    if not traj_path.exists():
        print(f"[错误] 轨迹文件不存在: {traj_path}", file=sys.stderr)
        return 2
    json_path = traj_path.with_suffix(".json")
    if not json_path.exists():
        print(f"[错误] 缺少同名元数据: {json_path}", file=sys.stderr)
        return 2

    import h5py
    from mani_skill.trajectory import utils as trajectory_utils
    from mani_skill.utils import io_utils

    meta = io_utils.load_json(str(json_path))
    env_info = meta["env_info"]
    env_id = env_info["env_id"]
    base_kwargs = dict(env_info.get("env_kwargs", {}))

    task_text = args.task_text or INSTRUCTIONS.get(env_id)
    if not task_text:
        print(f"[警告] 没有 {env_id} 的默认指令，请用 --task-text 指定", file=sys.stderr)

    print(f"[信息] 环境: {env_id}")
    print(f"[信息] 指令: {task_text}")
    print(f"[信息] 原始控制模式: {meta['episodes'][0].get('control_mode')}")

    env = build_env(env_id, base_kwargs, args.image_size, args.sim_backend)

    out_dir = Path(args.out).expanduser()
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    h5 = h5py.File(str(traj_path), "r")
    episodes = meta["episodes"]
    if args.count > 0:
        episodes = episodes[: args.count]
    if args.dry_run:
        episodes = episodes[:2]

    records = []
    state_dim = action_dim = None

    for i, ep in enumerate(episodes):
        ep_id = ep["episode_id"]
        traj_id = f"traj_{ep_id}"
        if traj_id not in h5:
            print(f"[跳过] {traj_id} 不在 h5 中")
            continue

        try:
            reset_kwargs = dict(ep.get("reset_kwargs", {}))
            seed = reset_kwargs.pop("seed", ep.get("episode_seed", 0))
            env.reset(seed=seed, **reset_kwargs)

            env_states = trajectory_utils.dict_to_list_of_dicts(h5[traj_id]["env_states"])
            env.unwrapped.set_state_dict(env_states[0])

            actions = np.asarray(h5[traj_id]["actions"][:], dtype=np.float32)
            if args.max_steps > 0:
                actions = actions[: args.max_steps]

            frames, states = [], []
            for t, a in enumerate(actions):
                img = env.render()
                frames.append(to_hwc_uint8(img))
                states.append(get_proprio(env))
                env.step(a)

            frames = np.stack(frames, axis=0)
            state_dim = int(states[0].shape[0])
            states_arr = (
                np.stack(states, axis=0)
                if state_dim > 0
                else np.zeros((len(actions), 0), dtype=np.float32)
            )
            action_dim = int(actions.shape[1])

            if not args.dry_run:
                ep_dir = out_dir / f"ep_{i:06d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                np.save(ep_dir / "frames.npy", frames)
                np.save(ep_dir / "states.npy", states_arr)
                np.save(ep_dir / "actions.npy", actions)

            records.append(
                {
                    "index": i,
                    "source_episode_id": int(ep_id),
                    "num_frames": int(frames.shape[0]),
                    "dir": f"ep_{i:06d}",
                }
            )
            print(f"  [{i + 1}/{len(episodes)}] ep_{i:06d}: {frames.shape[0]} 帧, "
                  f"图 {frames.shape[1]}x{frames.shape[2]}, 动作 {action_dim} 维")

        except Exception:
            print(f"[失败] episode {ep_id} 回放异常，已跳过：", file=sys.stderr)
            traceback.print_exc()
            continue

    h5.close()
    try:
        env.close()
    except Exception:
        pass

    if not records:
        print("[错误] 没有任何 episode 成功导出", file=sys.stderr)
        return 1

    if action_dim != 7:
        print(
            f"[警告] 动作维度是 {action_dim}，而 RPD 期望 7 维 (Δx, Δω, gripper)。\n"
            f"        请确认演示是否用 pd_ee_delta_pose 录制；否则需要在微调时对齐。",
            file=sys.stderr,
        )

    manifest = {
        "env_id": env_id,
        "task_text": task_text,
        "fps": args.fps,
        "image_size": args.image_size,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "control_mode": "pd_ee_delta_pose",
        "camera": "human",
        "total_episodes": len(records),
        "total_frames": int(sum(r["num_frames"] for r in records)),
        "episodes": records,
    }

    if args.dry_run:
        print("\n[dry-run] 未落盘。manifest 预览：")
        print(json.dumps({k: v for k, v in manifest.items() if k != "episodes"}, ensure_ascii=False, indent=2))
    else:
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[完成] {len(records)} 个 episode，共 {manifest['total_frames']} 帧 → {out_dir}")
        print(f"        下一步：用 lerobot 环境跑 stage2_pack_lerobot.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
