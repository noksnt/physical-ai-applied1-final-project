"""学習済みPPOのPickCube-v1動作を、摩擦条件を指定して1試行録画する"""

import argparse
from copy import deepcopy

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - PickCube-v1の環境登録
import sapien
import torch
from evaluate import load_policy
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils.wrappers.record import RecordEpisode
from paths import VIDEO_DIR


def record(gripper_friction, cube_friction, seed):
    # 既存のPanda設定を変更せず、Gripper摩擦を設定する
    @register_agent(override=True)
    class FrictionPanda(Panda):
        uid = "panda"
        urdf_config = deepcopy(Panda.urdf_config)

    gripper = FrictionPanda.urdf_config["_materials"]["gripper"]
    gripper["static_friction"] = gripper_friction
    gripper["dynamic_friction"] = gripper_friction

    output_dir = VIDEO_DIR / f"seed_{seed}" / f"gripper_{gripper_friction:g}_cube_{cube_friction:g}"
    if output_dir.exists():
        raise FileExistsError(f"出力先が既にあります: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(device)

    env = gym.make(
        "PickCube-v1",
        robot_uids="panda",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        sim_backend="physx_cpu",
        render_backend="cpu",
        render_mode="rgb_array",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    env = RecordEpisode(
        env,
        output_dir=str(output_dir),
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

        # Cubeの衝突形状に摩擦マテリアルを適用する。
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

        for step in range(100):
            with torch.no_grad():
                action = policy(obs.to(device))

            obs, _, terminated, truncated, info = env.step(action.to(base.device))
            steps = step + 1

            success = bool(info["success"].item())
            if success or bool(terminated.item()) or bool(truncated.item()):
                break

    finally:
        env.close()  # 動画を確定する。

    videos = sorted(output_dir.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"動画が保存されませんでした: {output_dir}")

    print(f"Success: {success}, Steps: {steps}")
    for video in videos:
        print(f"Video: {video}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gripper-friction", type=float, required=True)
    parser.add_argument("--cube-friction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=67)
    args = parser.parse_args()
    for value in (args.gripper_friction, args.cube_friction):
        if not 0 <= value <= 5:
            parser.error("摩擦係数は0〜5で指定してください")

    record(args.gripper_friction, args.cube_friction, args.seed)


if __name__ == "__main__":
    main()
