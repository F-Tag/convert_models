# convert yolox models to onnx to openvino
from pathlib import Path

import cv2
import numpy as np
import onnx
import openvino as ov
import torch
import torch.onnx
from nncf import CompressWeightsMode, compress_weights
from onnxconverter_common import auto_convert_mixed_precision, float16
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxsim import simplify
from tqdm import tqdm

NUM_INFERENCES = 50

MODEL_PATH = Path("models/midas")
OUTPUT_PATH = Path("outputs/midas")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

OPSET = 17


def main():

    model_name = "MiDaS_small"
    # model_name = "DPT_SwinV2_T_256"
    # model_name = "DPT_LeViT_224"
    model = torch.hub.load("isl-org/MiDaS", model_name)
    transforms = torch.hub.load("isl-org/MiDaS", "transforms")
    transform = transforms.small_transform
    # transform = transforms.swin256_transform
    # transform = transforms.levit_transform
    model.eval()
    model.to("cpu")

    # image load
    image = cv2.imread("dataset/5418492198_2f3aec2c44_o.jpg")
    batch = transform(image)

    torch.onnx.export(
        model,
        batch,
        MODEL_PATH / "base.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=OPSET,
    )

    # simplify
    model_onnx = onnx.load(MODEL_PATH / "base.onnx")
    for _ in range(5):
        model_onnx, check = simplify(model_onnx)
        if not check:
            break
    onnx.save(model_onnx, MODEL_PATH / "simple.onnx")

    # quantize
    quant_pre_process(
        MODEL_PATH / "simple.onnx",
        MODEL_PATH / "simple_q.onnx",
        skip_symbolic_shape=True,
    )
    quantize_dynamic(
        MODEL_PATH / "simple_q.onnx",
        MODEL_PATH / "simple_q.onnx",
        weight_type=QuantType.QUInt8,
    )

    # convert fp16
    model_simple = onnx.load(MODEL_PATH / "simple.onnx")
    model_fp16 = float16.convert_float_to_float16(model_simple, keep_io_types=True)
    # model_fp16 = auto_convert_mixed_precision(model_simple, batch, rtol=0.01, atol=0.001, keep_io_types=True)
    onnx.save(model_fp16, MODEL_PATH / "simple_fp16.onnx")

    # convert torch to openvino
    model_torch_openvino = ov.convert_model(model, example_input=batch)
    ov.save_model(model_torch_openvino, MODEL_PATH / "torch_openvino.xml")

    # convert onnx to openvino
    model_onnx_openvino = ov.convert_model(MODEL_PATH / "simple.onnx")
    ov.save_model(model_onnx_openvino, MODEL_PATH / "onnx_openvino.xml")

    # openvino quantize
    model_onnx_openvino_quantize = compress_weights(
        model_onnx_openvino,
        mode=CompressWeightsMode.INT4_SYM,
        group_size=64,
        ratio=0.9,
    )
    ov.save_model(
        model_onnx_openvino_quantize, MODEL_PATH / "onnx_openvino_quantize.xml"
    )

    print("running torch openvino inference ...")
    ov_core = ov.Core()
    model_torch_openvino = ov_core.read_model(MODEL_PATH / "torch_openvino.xml")
    compiled_model_torch_openvino = ov.compile_model(
        model_torch_openvino, "CPU", config={"INFERENCE_NUM_THREADS": 1}
    )
    output = compiled_model_torch_openvino(batch)
    draw_depth(OUTPUT_PATH / "torch_openvino.png", output)
    for i in tqdm(range(NUM_INFERENCES)):
        _ = compiled_model_torch_openvino(batch)

    print("running onnx openvino inference ...")
    model_onnx_openvino = ov_core.read_model(MODEL_PATH / "torch_openvino.xml")
    compiled_model_onnx_openvino = ov.compile_model(
        model_onnx_openvino, "CPU", config={"INFERENCE_NUM_THREADS": 1}
    )
    output = compiled_model_onnx_openvino(batch)
    draw_depth(OUTPUT_PATH / "onnx_openvino.png", output)
    for _ in tqdm(range(NUM_INFERENCES)):
        _ = compiled_model_onnx_openvino(batch)

    print("running onnx openvino quantize inference ...")
    model_onnx_openvino = ov_core.read_model(MODEL_PATH / "onnx_openvino_quantize.xml")
    compiled_model_onnx_openvino = ov.compile_model(
        model_onnx_openvino, "CPU", config={"INFERENCE_NUM_THREADS": 1}
    )
    output = compiled_model_onnx_openvino(batch)
    draw_depth(OUTPUT_PATH / "onnx_openvino_quantize.png", output)
    for _ in tqdm(range(NUM_INFERENCES)):
        _ = compiled_model_onnx_openvino(batch)

    print("running torch inference ...")
    with torch.no_grad():
        output = model(batch)
    draw_depth(OUTPUT_PATH / "torch.png", output)
    for _ in tqdm(range(NUM_INFERENCES)):
        with torch.no_grad():
            _ = model(batch)

    print("running onnxruntime inference ...")
    opts = SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    ort_sess = InferenceSession(MODEL_PATH / "base.onnx", opts)
    output = ort_sess.run(None, {"input": batch.numpy()})
    draw_depth(OUTPUT_PATH / "onnx.png", output)
    for i in tqdm(range(NUM_INFERENCES)):
        _ = ort_sess.run(None, {"input": batch.numpy()})

    print("running onnxruntime simplified inference ...")
    ort_sess = InferenceSession(MODEL_PATH / "simple.onnx", opts)
    output = ort_sess.run(None, {"input": batch.numpy()})
    draw_depth(OUTPUT_PATH / "onnx_simple.png", output)
    for i in tqdm(range(NUM_INFERENCES)):
        _ = ort_sess.run(None, {"input": batch.numpy()})

    print("running onnxruntime simplified quantized inference ...")
    ort_sess = InferenceSession(MODEL_PATH / "simple_q.onnx", opts)
    output = ort_sess.run(None, {"input": batch.numpy()})
    draw_depth(OUTPUT_PATH / "onnx_simple_q.png", output)
    for i in tqdm(range(NUM_INFERENCES)):
        _ = ort_sess.run(None, {"input": batch.numpy()})

    print("running onnxruntime simplified fp16 inference ...")
    ort_sess = InferenceSession(MODEL_PATH / "simple_fp16.onnx", opts)
    output = ort_sess.run(None, {"input": batch.numpy()})
    draw_depth(OUTPUT_PATH / "onnx_simple_fp16.png", output)
    for i in tqdm(range(NUM_INFERENCES)):
        _ = ort_sess.run(None, {"input": batch.numpy()})


def draw_depth(filepath, output):
    if isinstance(output, torch.Tensor):
        output = output.numpy()
    if not isinstance(output, np.ndarray):
        output = output[0]
    output = output.squeeze(0)
    output -= output.min()
    output /= output.max()
    output *= 255
    cv2.imwrite(str(filepath), output)


if __name__ == "__main__":
    main()
