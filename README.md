# ManiSkill3 → LeRobotDataset 转换工具

把 ManiSkill3 的演示轨迹转成 LeRobot 数据集格式，用于微调 SmolVLA（阶段 0）。

## 为什么是两段式

本机环境是分开的：

| 环境 | Python | 装了什么 |
|---|---|---|
| `rpd` | 3.11 | `mani_skill`、`sapien`、`h5py` |
| `lerobot` | 3.12 | `lerobot` 0.5.2（editable，指向 `~/lerobot/src`） |

ManiSkill3 和 LeRobot 的依赖（尤其 torch / numpy 版本）冲突风险高，**不建议合并环境**。
所以拆成两段，各自用自己环境跑：

```
Stage 1 (rpd)     回放轨迹 + 渲染 human camera  → 中间产物（npy）
Stage 2 (lerobot) 中间产物 → LeRobotDataset     （parquet + mp4 + meta）
```

`convert.sh` 会自动调用两个解释器，你只需要跑一条命令。

## 快速开始

```bash
cd ~/lerobot/tool
chmod +x convert.sh          # 首次

# 先干跑，确认能建环境、能取到图
./convert.sh \
  --traj ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --name pickcube --count 2 --dry-run

# 正式转换
./convert.sh \
  --traj ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --name pickcube --count 100 --verify
```

产物默认落在：

- 数据集：`~/datasets/lerobot/pickcube`
- 中间产物：`~/datasets/intermediate/pickcube`（转换成功后自动删除）

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--traj` | 必需 | 演示 `.h5`，同名 `.json` 必须存在 |
| `--name` | 必需 | 数据集短名，决定路径与 repo_id |
| `--count` | 全部 | 处理多少个 episode |
| `--fps` | 20 | 写入 meta 的帧率，需与演示一致 |
| `--image-size` | 256 | 与 RPD 的 `human_render_camera_configs` 一致 |
| `--intermediate` | `~/datasets/intermediate/<name>` | 中间产物目录 |
| `--root` | `~/datasets/lerobot/<name>` | 数据集落盘目录 |
| `--repo-id` | `local/<name>` | LeRobot 数据集标识 |
| `--dry-run` | — | 只回放 2 个 episode，不落盘 |
| `--verify` | — | 打包后重新加载自检 |
| `--keep-intermediate` | — | 保留中间产物 |

## 转换出来的数据集长什么样

```
~/datasets/lerobot/pickcube/
├── meta/
│   ├── info.json          # 特征定义、帧率、总数
│   ├── episodes/          # episode 索引
│   ├── tasks.jsonl        # 语言指令
│   └── stats.json         # 归一化统计量
├── data/chunk-000/*.parquet
└── videos/chunk-000/observation.images.human/*.mp4
```

特征：

| 键 | dtype | shape |
|---|---|---|
| `observation.images.human` | video | (3, 256, 256) |
| `observation.state` | float32 | (dof,) |
| `action` | float32 | (7,) |

## 关键设计决策

**① 渲染用 `human_render_camera_configs`，不是 `sensor_configs`**

RPD 源码里就是这么建的：

```python
env_kwargs = dict(
    obs_mode="rgb",
    control_mode="pd_ee_delta_pose",
    human_render_camera_configs={"width": 256, "height": 256},
)
```

用同一路相机，渲染出的画面才和 RPD 训练学生时看到的完全一致。

**② 控制模式固定 `pd_ee_delta_pose`**

这是 7 维动作 `(Δx, Δω, gripper)` 的来源，也是 RPD 和 OpenVLA 的 `maniskill_human:7.0.0` 对齐的前提。

**③ 语言指令与 RPD 源码逐字一致**

`stage1_replay_render.py` 里的 `INSTRUCTIONS` 字典是从
`RPD/src/rpd/ppo_rgb_rpd.py` 抄过来的。**改一个字都会导致指令嵌入对不上**，
所以不要随手改；需要自定义时用 `--task-text` 覆盖。

**④ 状态取关节位置 `qpos`**

`obs_mode="rgb"` 本身不含本体状态，所以直接从 `agent.robot.get_qpos()` 取。
SmolVLA 会把它 padding 到 32 维，维度不敏感。

## 常见问题

**Q: 报 `找不到解释器`**
用环境变量覆盖：

```bash
RPD_PY=/path/to/rpd/python LR_PY=/path/to/lerobot/python ./convert.sh ...
```

**Q: 报 `轨迹文件不存在`，但路径看起来是对的**
ManiSkill3 的演示默认在 `~/.maniskill/demos/<env_id>/` 下，按生成方式分子目录
（`motionplanning` / `rl`）。先确认文件真的下载了：

```bash
find ~/.maniskill -name "*.h5" | head
```

没下载过的话，ManiSkill3 首次运行时会自动拉取，或者手动
`python -m mani_skill.utils.download_demo <env_id>`。

**Q: 动作维度不是 7**
Stage 1 会打警告。说明原始演示不是用 `pd_ee_delta_pose` 录的。
要么换用 `pd_ee_delta_pose` 录制的演示，要么在微调时把动作维度对齐。

**Q: 显存 / 内存不够**
Stage 1 用 `sim_backend=cpu` 单环境回放，显存需求很低，但**中间产物的 npy 很占磁盘**：
每帧 256×256×3 = 192KB，1000 个 episode × 100 步 ≈ 19GB。
用 `--count` 控制规模，或转换完让它自动删除中间产物。

**Q: 想跳过 MP4 直接存 PNG 调试**
`stage2_pack_lerobot.py --no-videos`。体积会大很多，只适合排查问题。

## 已知限制

- 只支持单环境回放（`num_envs=1`），速度受渲染帧率限制，几百个 episode 需要一些耐心
- 不做控制模式转换。若演示是 `pd_joint_pos` 录的，需要先自己转成 `pd_ee_delta_pose`
- `observation.state` 只取关节位置，不含末端位姿 / 夹爪宽度等派生量
- 不处理多相机。RPD 只用一路 human camera，够用
