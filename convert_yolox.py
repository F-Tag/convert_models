# convert yolox models to onnx to openvino
import onnx
import openvino as ov
import torch
import torch.onnx
from onnxruntime import InferenceSession, SessionOptions
from onnxsim import simplify
from tqdm import tqdm

NUM_INFERENCES = 200


def main():

    model = torch.hub.load("Megvii-BaseDetection/YOLOX", "yolox_tiny")
    model.eval()
    model.to("cpu")

    dummy_input = torch.randn(1, 3, 416, 416)

    torch.onnx.export(
        model,
        dummy_input,
        "yolox_tiny.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )

    # simplify
    model_onnx = onnx.load("yolox_tiny.onnx")
    for _ in range(5):
        model_onnx, check = simplify(model_onnx)
        if not check:
            break
    onnx.save(model_onnx, "yolox_tiny_simpl.onnx")

    # convert torch to openvino
    model_torch_openvino = ov.convert_model(model, example_input=dummy_input)
    ov.save_model(model_torch_openvino, "yolox_tiny_torch_openvino.xml")

    # convert onnx to openvino
    model_onnx_openvino = ov.convert_model("yolox_tiny_simpl.onnx")
    ov.save_model(model_onnx_openvino, "yolox_tiny_onnx_openvino.xml")

    print("running torch openvino inference ...")
    ov_core = ov.Core()
    model_torch_openvino = ov_core.read_model("yolox_tiny_torch_openvino.xml")
    compiled_model_torch_openvino = ov.compile_model(
        model_torch_openvino, "CPU", config={"INFERENCE_NUM_THREADS": 1}
    )
    for i in tqdm(range(NUM_INFERENCES)):
        _ = compiled_model_torch_openvino(dummy_input)

    print("running onnx openvino inference ...")
    model_onnx_openvino = ov_core.read_model("yolox_tiny_torch_openvino.xml")
    compiled_model_onnx_openvino = ov.compile_model(
        model_onnx_openvino, "CPU", config={"INFERENCE_NUM_THREADS": 1}
    )
    for _ in tqdm(range(NUM_INFERENCES)):
        _ = compiled_model_onnx_openvino(dummy_input)

    print("running torch inference ...")
    for _ in tqdm(range(NUM_INFERENCES)):
        with torch.no_grad():
            _ = model(dummy_input)

    print("running onnxruntime inference ...")
    opts = SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    ort_sess = InferenceSession("yolox_tiny.onnx", opts)
    for i in tqdm(range(NUM_INFERENCES)):
        _ = ort_sess.run(None, {"input": dummy_input.numpy()})

    print("running onnxruntime simplified inference ...")
    ort_sess = InferenceSession("yolox_tiny_simpl.onnx", opts)
    for i in tqdm(range(NUM_INFERENCES)):
        _ = ort_sess.run(None, {"input": dummy_input.numpy()})


if __name__ == "__main__":
    main()
