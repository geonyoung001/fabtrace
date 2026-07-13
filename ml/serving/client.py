# ml/serving/client.py
"""Triton(wafer_classifier)로 실제 wafer map을 추론하는 검증용 클라이언트.

test set에서 몇 장을 꺼내 dataset.py와 동일한 전처리(resize + one-hot)를 적용하고,
HTTP(8000)로 Triton에 보내 예측 클래스/확신도를 정답과 비교 출력한다.

실행:  python -m ml.serving.client   (Triton이 localhost:8000 에 떠 있어야 함)
"""
import argparse

import numpy as np
import tritonclient.http as httpclient

from ml.config import IDX_TO_LABEL, PROCESSED_DATA_DIR, TRITON_MODEL_NAME
from ml.train.dataset import WM811KDataset

URL = "localhost:8000"


def softmax(logits: np.ndarray) -> np.ndarray:
    # 행(row)별 안정적 softmax
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main(n_samples: int):
    # 1) 서버 상태 확인
    client = httpclient.InferenceServerClient(url=URL)
    try:
        server_ready = client.is_server_ready()
    except (ConnectionRefusedError, OSError, Exception) as e:
        raise SystemExit(
            f"[!] Triton 서버에 연결할 수 없음 ({URL}): {type(e).__name__}\n"
            f"    docker run --gpus all ... tritonserver 로 먼저 서버를 띄우세요."
        )
    if not server_ready:
        raise SystemExit(f"[!] Triton 서버가 준비되지 않음 ({URL}).")
    if not client.is_model_ready(TRITON_MODEL_NAME):
        raise SystemExit(f"[!] 모델 '{TRITON_MODEL_NAME}' 이 READY 상태가 아님.")

    # 2) test set 로드 + dataset.py와 동일 전처리
    maps = np.load(PROCESSED_DATA_DIR / "X_test.npy", allow_pickle=True)
    true_labels = np.load(PROCESSED_DATA_DIR / "y_test.npy", allow_pickle=True)  # 문자열 라벨

    n = min(n_samples, len(maps))
    # labels 인자는 전처리에 쓰이지 않으므로 더미(0)로 채움
    ds = WM811KDataset(maps[:n], np.zeros(n, dtype=np.int64), augment=False)
    batch = np.stack([ds[i][0].numpy() for i in range(n)]).astype(np.float32)  # (n, 3, 64, 64)

    # 3) Triton 추론 요청
    inp = httpclient.InferInput("input", batch.shape, "FP32")
    inp.set_data_from_numpy(batch)
    out = httpclient.InferRequestedOutput("logits")
    resp = client.infer(TRITON_MODEL_NAME, inputs=[inp], outputs=[out])

    logits = resp.as_numpy("logits")          # (n, 9)
    probs = softmax(logits)
    preds = probs.argmax(axis=1)

    # 4) 결과 비교 출력
    correct = 0
    print(f"\n{'idx':>3} | {'true':<10} | {'pred':<10} | {'conf':>6} | ok")
    print("-" * 44)
    for i in range(n):
        true = str(true_labels[i])
        pred = IDX_TO_LABEL[int(preds[i])]
        ok = (true == pred)
        correct += ok
        print(f"{i:>3} | {true:<10} | {pred:<10} | {probs[i, preds[i]]:>5.1%} | {'✓' if ok else '✗'}")
    print("-" * 44)
    print(f"accuracy on {n} samples: {correct}/{n} = {correct / n:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-samples", type=int, default=10, help="추론할 test 샘플 수")
    args = parser.parse_args()
    main(args.n_samples)
