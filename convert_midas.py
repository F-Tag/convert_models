# convert yolox models to onnx to openvino
from argparse import ArgumentParser
from pathlib import Path

import cv2
import onnx
import torch
import torch.onnx
from onnxruntime import InferenceSession, SessionOptions, get_available_providers
from onnxsim import simplify
from tqdm import tqdm

from convert_models.midas.vis import draw_depth

MODEL_PATH = Path("models/midas")
OUTPUT_PATH = Path("outputs/midas")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def main():

    parser = ArgumentParser("convert midas models to onnx")
    parser.add_argument(
        "--num-inferences", type=int, default=200, help="number of inferences"
    )
    parser.add_argument(
        "--provider", type=str, default="CPUExecutionProvider", help="provider"
    )
    parser.add_argument(
        "--num-simplify", type=int, default=10, help="number of simplify"
    )
    parser.add_argument("--opset", type=int, default=11, help="opset")
    args = parser.parse_args()

    # check provider
    assert (
        args.provider in get_available_providers()
    ), f"{args.provider} not found in {get_available_providers()}"

    # load MiDaS model
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
    image_path = "dataset/5418492198_2f3aec2c44_o.jpg"
    image_path = "dataset/14028460030_79f3be107e_o.jpg"
    image = cv2.imread(image_path)
    assert image is not None, f"{image_path} not found"
    batch = transform(image)

    torch.onnx.export(
        model,
        batch,
        MODEL_PATH / "base.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=args.opset,
    )

    # simplify
    model_onnx = onnx.load(MODEL_PATH / "base.onnx")
    for _ in range(args.num_simplify):
        model_onnx, check = simplify(model_onnx)
        if not check:
            break
    onnx.save(model_onnx, MODEL_PATH / "simple.onnx")

    # Run inference with torch
    print("running torch inference ...")
    with torch.no_grad():
        output = model(batch)
    draw_depth(OUTPUT_PATH / "torch.png", output)
    for _ in tqdm(range(args.num_inferences)):
        with torch.no_grad():
            _ = model(batch)

    # Run inference with onnxruntime
    print("running onnxruntime inference ...")
    opts = SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    ort_sess = InferenceSession(
        MODEL_PATH / "base.onnx", opts, providers=["CPUExecutionProvider"]
    )
    output = ort_sess.run(None, {"input": batch.numpy()})
    draw_depth(OUTPUT_PATH / "onnx.png", output)
    for i in tqdm(range(args.num_inferences)):
        _ = ort_sess.run(None, {"input": batch.numpy()})

    # Run inference with onnxruntime simplified
    print("running onnxruntime simplified inference ...")
    ort_sess = InferenceSession(
        MODEL_PATH / "simple.onnx", opts, providers=["CPUExecutionProvider"]
    )
    output = ort_sess.run(None, {"input": batch.numpy()})
    draw_depth(OUTPUT_PATH / "onnx_simple.png", output)
    for i in tqdm(range(args.num_inferences)):
        _ = ort_sess.run(None, {"input": batch.numpy()})

    if args.provider == "CPUExecutionProvider":
        return
    elif args.provider == "OpenVINOExecutionProvider":
        # Run inference with openvino
        print(
            "running onnxruntime simplified with OpenVINOExecutionProvider inference ..."
        )
        ort_sess = InferenceSession(
            MODEL_PATH / "simple.onnx",
            opts,
            providers=["OpenVINOExecutionProvider"],
            provider_options=[{"num_of_threads": 1}],
        )
        output = ort_sess.run(None, {"input": batch.numpy()})
        draw_depth(OUTPUT_PATH / "onnx_simple_openvion_exec.png", output)
        for i in tqdm(range(args.num_inferences)):
            _ = ort_sess.run(None, {"input": batch.numpy()})

    elif args.provider == "CUDAExecutionProvider":
        print("running onnxruntime simplified with CUDAExecutionProvider inference ...")
        ort_sess = InferenceSession(
            MODEL_PATH / "simple.onnx",
            opts,
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": 0}],
        )
        output = ort_sess.run(None, {"input": batch.numpy()})
        draw_depth(OUTPUT_PATH / "onnx_simple_cuda_exec.png", output)
        for i in tqdm(range(args.num_inferences)):
            _ = ort_sess.run(None, {"input": batch.numpy()})

    elif args.provider == "TensorrtExecutionProvider":
        print(
            "running onnxruntime simplified with TensorrtExecutionProvider inference ..."
        )
        ort_sess = InferenceSession(
            MODEL_PATH / "simple.onnx",
            opts,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
            provider_options=[{"device_id": 0}],
        )
        output = ort_sess.run(None, {"input": batch.numpy()})
        draw_depth(OUTPUT_PATH / "onnx_simple_tensorrt_exec.png", output)
        for i in tqdm(range(args.num_inferences)):
            _ = ort_sess.run(None, {"input": batch.numpy()})


if __name__ == "__main__":
    main()
