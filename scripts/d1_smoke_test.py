"""Smoke test：验证 smolvla_base 在本地数据上完成一次前向。

覆盖三段：读取数据集 -> 构建 policy 与预/后处理器 -> select_action。
目的是在长时间训练之前暴露配置不匹配类错误，主要是相机键位与归一化统计量。

Usage:
    python scripts/d1_smoke_test.py

Output:
    成功时打印预测动作形状 (1, 6) 及与真值的对照，并以 "D1 SMOKE TEST PASSED" 结束。

Notes:
    - 模型声明 3 个相机位（camera1/2/3），数据集只提供 up/side；
      RENAME 需把数据集键映射到 camera1/2，第三位留空即可通过子集校验。
    - 两种调用形态都会尝试：单帧、带 batch 维，以覆盖不同的调用习惯。
"""
import os, sys, traceback
from pathlib import Path
import torch

# 本地权重副本由 fetch_model_modelscope.py 下载；改为 repo id 会走慢速镜像（~175 KB/s）
MODEL = str(Path(__file__).resolve().parents[1] / "data/models/smolvla_base")
# NOTE: 映射目标必须是 camera1/camera2；第三位 camera3 不提供，校验按子集处理
RENAME = {"observation.images.up": "observation.images.camera1",
          "observation.images.side": "observation.images.camera2"}

print("=" * 62)
# ---- [0] 先确认环境：torch 版本 + CUDA 是否真的可用 ----
print("[0] 环境  torch:", torch.__version__, "| cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("    相机重命名映射:", RENAME)

print("=" * 62)
# ---- [1] 读数据集：确认 episodes 数、任务字符串 ----
print("[1] 读公开数据集 lerobot/svla_so101_pickplace")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("lerobot/svla_so101_pickplace")
print("    episodes:", ds.num_episodes, "| frames:", ds.num_frames, "| fps:", ds.fps)
sample = ds[0]
print("    任务字符串:", repr(sample.get("task")))

print("=" * 62)
# ---- [2] 加载模型：看参数量、默认冻结策略、动作块长度 ----
print("[2] 加载基座模型（本地 ModelScope 副本）")
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

cfg = SmolVLAConfig.from_pretrained(MODEL)
cfg.pretrained_path = MODEL
print("    device:", cfg.device, "| chunk_size:", cfg.chunk_size,
      "| n_action_steps:", cfg.n_action_steps)
print("    freeze_vision_encoder:", cfg.freeze_vision_encoder,
      "| train_expert_only:", cfg.train_expert_only)
print("    scheduler_decay_steps:", cfg.scheduler_decay_steps, "| lr:", cfg.optimizer_lr)

policy = make_policy(cfg, ds_meta=ds.meta, rename_map=RENAME)
print("    策略参数量: %.1f M" % (sum(p.numel() for p in policy.parameters()) / 1e6))
DEV = next(policy.parameters()).device
print("    设备:", DEV)
policy.eval()

pre, post = make_pre_post_processors(cfg, pretrained_path=MODEL)
print("    预处理器就绪")

print("=" * 62)
# ---- [3] 前向：先试单帧，失败再试带 batch 维（两种调用习惯都覆盖）----
print("[3] select_action 冒烟测试")
obs_raw = {k: v for k, v in sample.items() if k not in ("action", "action_is_pad")}
obs = {}
for k, v in obs_raw.items():
    obs[RENAME.get(k, k)] = v

def run_once(o):
    batch = pre(o)
    batch = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        return policy.select_action(batch)

cases = [("单帧", obs),
         ("带 batch 维度",
          {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in obs.items()})]

for tag, o in cases:
    try:
        act = run_once(o)
        print("  [OK] %s 方式成功" % tag)
        print("  预测动作 shape:", tuple(act.shape))
        print("  预测动作:", [round(float(x), 4) for x in act.flatten()[:8]])
        print("  真值动作:", [round(float(x), 4) for x in sample["action"].flatten()[:8]])
        print("=" * 62)
        print("D1 SMOKE TEST PASSED")
        sys.exit(0)
    except Exception as e:
        print("  [FAIL] %s 方式: %s: %s" % (tag, type(e).__name__, str(e)[:300]))

print("=" * 62)
print("D1 SMOKE TEST FAILED")
