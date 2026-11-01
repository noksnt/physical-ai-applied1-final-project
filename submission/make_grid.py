"""同じseedのGripper摩擦4条件を、同期した2x2比較動画にまとめる"""

import argparse
import csv
import math
import re
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
from paths import CSV_DIR, GRID_DIR, VIDEO_DIR
from PIL import Image, ImageDraw, ImageFont

FRICTIONS = (2.0, 0.5, 0.1, 0.01)
FPS = 10
WIDTH, HEIGHT = 640, 480


def video_duration(ffmpeg, path):
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True
    )

    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise RuntimeError(f"動画の長さを取得できません: {path}")

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def success_label(gripper, cube, seed):
    name = f"gripper_{gripper:g}_cube_{cube:g}_seed_{seed}.csv"
    path = CSV_DIR / name

    if not path.is_file():
        return "NO CSV"

    with path.open(newline="") as file:
        value = next(csv.DictReader(file))["success"].strip().lower()

    if value == "true":
        return "SUCCESS"
    if value == "false":
        return "FAILURE"
    raise ValueError(f"不正な成功判定: {value}")


def make_label(path, gripper, status):
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 54), fill=(0, 0, 0, 190))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 25)
    except OSError:
        font = ImageFont.load_default()

    color = {"SUCCESS": "#8FE0A8", "FAILURE": "#FF9999"}.get(status, "white")
    draw.text((14, 12), f"Gripper {gripper:g}", font=font, fill="white")
    draw.text((380, 12), status, font=font, fill=color)
    image.save(path)


def make_grid(seed, cube):
    videos = [
        VIDEO_DIR / f"seed_{seed}" / f"gripper_{gripper:g}_cube_{cube:g}" / "0.mp4"
        for gripper in FRICTIONS
    ]
    for video in videos:
        if not video.is_file():
            raise FileNotFoundError(f"録画が見つかりません: {video}")

    output = GRID_DIR / f"seed_{seed}_cube_{cube:g}_grid.mp4"
    if output.exists():
        raise FileExistsError(f"出力先が既にあります: {output}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    duration = math.ceil(max(video_duration(ffmpeg, v) for v in videos) * FPS) / FPS
    statuses = [success_label(g, cube, seed) for g in FRICTIONS]

    with tempfile.TemporaryDirectory() as tmp:
        labels = [Path(tmp) / f"label_{i}.png" for i in range(4)]
        for label, gripper, status in zip(labels, FRICTIONS, statuses, strict=False):
            make_label(label, gripper, status)

        command = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-n"]
        for video in videos:
            command += ["-i", str(video)]
        for label in labels:
            command += ["-loop", "1", "-i", str(label)]

        filters = []
        for i in range(4):
            filters.append(
                f"[{i}:v]fps={FPS},scale={WIDTH}:{HEIGHT}:"
                "force_original_aspect_ratio=decrease,"
                f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,"
                "setsar=1,setpts=PTS-STARTPTS,"
                f"tpad=stop_mode=clone:stop_duration={duration},"
                f"trim=duration={duration},setpts=PTS-STARTPTS[base{i}]"
            )
            filters.append(f"[base{i}][{i + 4}:v]overlay=0:0:shortest=1[v{i}]")
        filters.append(
            "[v0][v1][v2][v3]xstack=inputs=4:"
            "layout=0_0|w0_0|0_h0|w0_h0:fill=black,"
            "format=yuv420p[out]"
        )
        command += [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-r",
            str(FPS),
            "-t",
            str(duration),
            "-movflags",
            "+faststart",
            str(output),
        ]
        GRID_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)

    for gripper, status in zip(FRICTIONS, statuses, strict=False):
        print(f"Gripper {gripper:g}: {status}")
    print(f"保存: {output} ({duration:g}秒)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cube-friction", type=float, default=0.1)
    args = parser.parse_args()

    make_grid(args.seed, args.cube_friction)


if __name__ == "__main__":
    main()
