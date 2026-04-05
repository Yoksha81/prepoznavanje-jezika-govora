from pathlib import Path
import os

PROJECT_ROOT = Path(
    os.environ.get(
        "VLID_PROJECT_ROOT",
        Path(__file__).resolve().parent.parent,
    )
)

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
WORK_DIR = PROJECT_ROOT / "work"

FRAMES_DIR = WORK_DIR / "frames"
FRAMES_A_DIR = FRAMES_DIR / "A"
FRAMES_L_DIR = FRAMES_DIR / "L"
FRAMES_R_DIR = FRAMES_DIR / "R"
FRAMES_BY_CAMERA = {
    "A": FRAMES_A_DIR,
    "L": FRAMES_L_DIR,
    "R": FRAMES_R_DIR,
}

MANIFESTS_DIR = WORK_DIR / "manifests"
FEATURES_DIR = WORK_DIR / "features"
RESULTS_DIR = WORK_DIR / "results"
DEBUG_DIR = WORK_DIR / "debug"

TUNING_DIR = RESULTS_DIR / "model_tuning"
FINAL_RESULTS_DIR = RESULTS_DIR / "final_models"

INPUT_MANIFEST = MANIFESTS_DIR / "manifest_all_konzistentan.csv"
CROP_MANIFEST = MANIFESTS_DIR / "crop_manifest_40f.csv"

VIDEO_EXT = ".mp4"
ROI_EXT = ".txt"
ALIGN_EXT = ".align"

TEST_SPEAKERS = {"spk04", "spk08", "spk14", "spk26", "spk28"}

COL_NAME = "name"
COL_LANGUAGE = "language"
COL_VIDEO_A = "video_a"
COL_VIDEO_L = "video_l"
COL_VIDEO_R = "video_r"
COL_COMMON = "common"
COL_TRANSCRIPT = "transcript"

ALIGN_SCALE = 10_000_000
N_FRAMES = 40
ROI_PADDING = 0

ROI_OUT_SIZE = (64, 64)
ROI_OUT_SIZE_BY_CAMERA = {
    "A": (112, 64),
    "L": (104, 64),
    "R": (88, 64),
}

HOG_PREFIX = "hog_static"
RESNET_PREFIX = "resnet18_meanstd"

HOG_ORIENTATIONS = 8
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
USE_DELTA_FEATURES = False

RESNET_INPUT_SIZE = 224
RESNET_USE_STD_AGG = True

CV_RESULTS_PATH = TUNING_DIR / "cv_results_all.csv"
BEST_PARAMS_PATH = TUNING_DIR / "best_params.csv"
BEST_PARAMS_SUMMARY_PATH = TUNING_DIR / "best_params_summary.csv"
