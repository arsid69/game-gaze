"""
One-time conversion: PyTorch L2CS-Net weights -> ONNX format.

Run this ONCE. It creates models/l2cs_gaze360.onnx, which the updated
milestone3 script will use instead of raw PyTorch for faster CPU inference.

Run: python export_to_onnx.py
"""

import pathlib
import torch
from l2cs import getArch

CWD = pathlib.Path.cwd()
PYTORCH_WEIGHTS = CWD / "models" / "L2CSNet_gaze360.pkl"
ONNX_OUTPUT = CWD / "models" / "l2cs_gaze360.onnx"
BINS = 90

model = getArch("ResNet50", BINS)
state_dict = torch.load(PYTORCH_WEIGHTS, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(1, 3, 448, 448)

# WARNING — these output NAMES are wrong, deliberately left as-is.
# L2CS-Net's forward() returns (pre_yaw_gaze, pre_pitch_gaze) — yaw FIRST —
# but this export labels tensor[0] "pitch_bins". The labels are only strings
# attached at export time; they do not change what the tensors contain.
# gaze_pipeline._predict_gaze_radians() reads tensor[0] as YAW, which is
# correct. If you ever "fix" the names here you MUST also change the unpack
# order there, or every gaze reading silently transposes its axes.
torch.onnx.export(
    model,
    dummy_input,
    str(ONNX_OUTPUT),
    input_names=["input"],
    output_names=["pitch_bins", "yaw_bins"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=12,
    dynamo=False,  # use the older, more stable exporter (avoids onnxscript dependency)
)

print(f"Exported ONNX model to: {ONNX_OUTPUT}")
