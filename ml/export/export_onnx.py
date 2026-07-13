# ml/export/export_onnx.py
# best_model.pt(state_dict) -> ONNX export
import torch

from ml.config import BEST_MODEL_PATH, INPUT_SHAPE, NUM_CLASSES, ONNX_MODEL_PATH
from ml.train.model import WaferClassifier

# Triton 24.01(server 2.42.0) 내장 onnxruntime이 지원하는 최대 ONNX IR version
TARGET_IR_VERSION = 9


def main():
    device = torch.device("cpu")  # export는 CPU로 충분

    # 모델 로드 (export 시에는 pretrained 다운로드 불필요)
    model = WaferClassifier(num_classes=NUM_CLASSES, pretrained=False)
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 더미 입력 (one-hot wafer map), batch 차원을 붙여 (1, 3, 64, 64)
    dummy_input = torch.randn(1, *INPUT_SHAPE, device=device)

    # Triton model repository의 version 디렉터리(<repo>/<model>/1/) 보장
    ONNX_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_MODEL_PATH),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"ONNX model exported to: {ONNX_MODEL_PATH}")

    # 검증 + 외부데이터 통합: onnx 로드/체크 후 단일 파일로 재저장
    # (dynamo exporter가 가중치를 model.onnx.data로 분리 저장하는데,
    #  Triton 배포 시 단일 model.onnx가 다루기 쉬움)
    try:
        import onnx
        onnx_model = onnx.load(str(ONNX_MODEL_PATH))  # 외부데이터까지 메모리로 로드
        onnx.checker.check_model(onnx_model)
        # IR version을 9로 낮춤: Triton 24.01 내장 onnxruntime이 IR<=9만 지원.
        # (최신 onnx는 IR 10으로 저장 → 구버전 런타임이 못 읽음. opset 18은 IR 9와 호환)
        if onnx_model.ir_version > TARGET_IR_VERSION:
            onnx_model.ir_version = TARGET_IR_VERSION
        # 가중치를 model.onnx 안에 임베드하여 재저장 → 분리된 .data 파일 제거
        onnx.save(onnx_model, str(ONNX_MODEL_PATH), save_as_external_data=False)
        ext_data = ONNX_MODEL_PATH.with_name(ONNX_MODEL_PATH.name + ".data")
        if ext_data.exists():
            ext_data.unlink()
        print("ONNX model check passed (weights embedded into single file)")
    except ImportError:
        print("onnx 미설치 → 구조 검증 건너뜀 (pip install onnx)")

    try:
        import numpy as np
        import onnxruntime as ort

        with torch.no_grad():
            torch_out = model(dummy_input).cpu().numpy()

        sess = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
        ort_out = sess.run(["logits"], {"input": dummy_input.cpu().numpy()})[0]

        max_diff = np.abs(torch_out - ort_out).max()
        print(f"PyTorch vs ONNXRuntime max diff: {max_diff:.6e}")
        assert np.allclose(torch_out, ort_out, atol=1e-4), "출력 불일치!"
        print("PyTorch/ONNXRuntime outputs match")
    except ImportError:
        print("onnxruntime 미설치 → 수치 검증 건너뜀 (pip install onnxruntime)")


if __name__ == "__main__":
    main()
