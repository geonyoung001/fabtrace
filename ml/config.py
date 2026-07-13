# ml/config.py
"""프로젝트 전역 상수 · 경로 정의 (single source of truth).

여러 스크립트에서 공유하는 클래스 목록, 입력 텐서 규격, 데이터/체크포인트 경로를
이 파일 한 곳에서만 정의한다. 다른 모듈은 값을 재정의하지 말고 여기서 import 한다.
"""
from pathlib import Path

# --- 경로 (repo 구조 기반으로 계산, 실행 위치(CWD)와 무관) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../fab_scope
ML_DIR = PROJECT_ROOT / "ml"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENTS_DATA_DIR = PROCESSED_DATA_DIR / "experiments"

CHECKPOINTS_DIR = ML_DIR / "checkpoints"
EXPERIMENTS_CHECKPOINT_DIR = CHECKPOINTS_DIR / "experiments"
EXPORT_DIR = ML_DIR / "export"

BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pt"

# --- Triton Inference Server ---
# model repository 레이아웃: <repo>/<model_name>/<version>/model.onnx
TRITON_MODEL_NAME = "wafer_classifier"
TRITON_MODEL_VERSION = "1"
MODEL_REPOSITORY_DIR = ML_DIR / "serving" / "model_repository"
# export는 Triton이 요구하는 위치·파일명(model.onnx)으로 바로 저장한다.
ONNX_MODEL_PATH = (MODEL_REPOSITORY_DIR / TRITON_MODEL_NAME /
                   TRITON_MODEL_VERSION / "model.onnx")

# --- 클래스 (WM811K failureType) ---
# 리스트 순서 = 정수 라벨 인덱스. WaferClassifier의 num_classes와 반드시 일치.
# (순서를 바꾸면 기존 체크포인트와 호환되지 않으므로 변경 금지)
FAILURE_CLASSES = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
                   'Loc', 'Near-full', 'Random', 'Scratch']
NUM_CLASSES = len(FAILURE_CLASSES)
LABEL_TO_IDX = {label: i for i, label in enumerate(FAILURE_CLASSES)}
IDX_TO_LABEL = {i: label for i, label in enumerate(FAILURE_CLASSES)}

# --- 입력 텐서 규격 ---
IMG_SIZE = 64        # resize 후 wafer map 한 변 길이
NUM_CHANNELS = 3     # one-hot 채널: [wafer 밖, 정상 die, 불량 die]
INPUT_SHAPE = (NUM_CHANNELS, IMG_SIZE, IMG_SIZE)  # (3, 64, 64), batch 차원 제외
