"""プロジェクトで使用するパス"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints/ppo_pd_ee_delta_pos_ckpt.pt"
CSV_DIR = ROOT / "results/submission_validation"
VIDEO_DIR = ROOT / "results/submission_videos"
GRID_DIR = ROOT / "results/submission_grids"
