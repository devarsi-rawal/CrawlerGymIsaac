import numpy as np
import onnx
import onnxruntime as ort
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Specify onnx file")
parser.add_argument("--input", type=float, nargs="+", help="Specify list inputs")
args = parser.parse_args()

fn = args.model
onnx_model = onnx.load(fn)
print(onnx.checker.check_model(onnx_model))

ort_model = ort.InferenceSession(fn)

print(args.input)

outputs = ort_model.run(None,
        {"obs": np.array([args.input]).astype(np.float32)}
        )

print(outputs)
