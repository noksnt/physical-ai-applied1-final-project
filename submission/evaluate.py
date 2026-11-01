"""学習済みPPOでPickCube-v1を評価する（1回の実行で1試行）"""

import argparse
import csv
from copy import deepcopy

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - PickCube-v1の環境登録
import sapien
import torch
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils.wrappers.record import RecordEpisode
from paths import CHECKPOINT, CSV_DIR, VIDEO_DIR
from torch import nn


class Actor(nn.Module):
    """学習済みチェックポイントと同じ構造"""

    def __init__(self):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(42, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 4),
        )

    def forward(self, obs):
        return self.actor_mean(obs)


def load_policy(device):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    weights = {k: v for k, v in checkpoint.items() if k.startswith("actor_mean.")}
    policy = Actor().to(device)
    policy.load_state_dict(weights, strict=True)
    return policy.eval()  # 評価モードへの切り替え


def evaluate(gripper_friction, cube_friction, seed, record=False):
    csv_path = CSV_DIR / f"gripper_{gripper_friction:g}_cube_{cube_friction:g}_seed_{seed}.csv"
    video_dir = VIDEO_DIR / f"seed_{seed}" / f"gripper_{gripper_friction:g}_cube_{cube_friction:g}"
    if csv_path.exists():
        raise FileExistsError(f"既存CSVを上書きしません: {csv_path}")
    if record and video_dir.exists():
        raise FileExistsError(f"既存動画フォルダを上書きしません: {video_dir}")

    # Pandaの設定をコピーし、指先の静止摩擦・動摩擦を変更する
    @register_agent(override=True)
    class FrictionPanda(Panda):
        uid = "panda"
        urdf_config = deepcopy(Panda.urdf_config)

    gripper = FrictionPanda.urdf_config["_materials"]["gripper"]
    gripper["static_friction"] = gripper_friction
    gripper["dynamic_friction"] = gripper_friction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(device)

    env = gym.make(
        "PickCube-v1",  # Cubeを掴んで目標位置へ移すタスク
        robot_uids="panda",  # ロボットの種類
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        sim_backend="physx_cpu",
        render_backend="cpu" if record else "none",
        **({"render_mode": "rgb_array"} if record else {}),
    )
    if record:
        video_dir.mkdir(parents=True, exist_ok=False)
        env = RecordEpisode(
            env,
            output_dir=str(video_dir),
            save_trajectory=False,
            save_video=True,
            video_fps=10,
            info_on_video=False,
        )

    success = False
    steps = 0
    try:
        obs, _ = env.reset(seed=seed)
        base = env.unwrapped
        assert isinstance(base.agent, FrictionPanda)

        # Cubeのすべての衝突形状に指定の摩擦を設定する
        material = sapien.physx.PhysxMaterial(
            static_friction=cube_friction,
            dynamic_friction=cube_friction,
            restitution=0.0,
        )

        shape_count = 0
        for entity in base.cube._objs:
            body = entity.find_component_by_type(sapien.physx.PhysxRigidBodyComponent)
            if body is None:
                raise RuntimeError("Cube rigid body not found")

            for shape in body.get_collision_shapes():
                shape.set_physical_material(material)
                actual = shape.get_physical_material()
                assert abs(actual.static_friction - cube_friction) < 1e-5
                assert abs(actual.dynamic_friction - cube_friction) < 1e-5

                shape_count += 1

        assert shape_count > 0

        # ロボットを動かす
        for step in range(100):
            with torch.no_grad():
                action = policy(obs.to(device))

            obs, _, terminated, truncated, info = env.step(action.to(base.device))
            steps = step + 1

            success = bool(info["success"].item())
            if success or bool(terminated.item()) or bool(truncated.item()):
                break

    finally:
        env.close()

    result = dict(
        gripper_friction=gripper_friction,
        cube_friction=cube_friction,
        seed=seed,
        success=success,
        steps=steps,
    )

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="") as f:  # 既存ファイルを上書きしない
        writer = csv.DictWriter(f, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)

    print("RESULT:", result)
    print("Saved:", csv_path)
    if record:
        print("Video directory:", video_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gripper-friction", type=float, required=True)
    parser.add_argument("--cube-friction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--record", action="store_true", help="評価と同時に動画も記録する")
    args = parser.parse_args()
    for name, value in (("gripper", args.gripper_friction), ("cube", args.cube_friction)):
        if not 0 <= value <= 5:
            parser.error(f"{name} friction must be between 0 and 5")

    evaluate(args.gripper_friction, args.cube_friction, args.seed, record=args.record)


if __name__ == "__main__":
    main()
