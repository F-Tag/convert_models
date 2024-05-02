import cv2
import numpy as np
import torch


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
