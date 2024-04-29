import numpy as np
import cv2
import os

OUT_DIR = "out"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


def make_video_from_array(arr: np.array, fname: str, fps: int):
    # switch blue and red channels
    arr = arr[..., ::-1]
    dims = arr.shape[2], arr.shape[1]
    path = os.path.join(OUT_DIR, f"{fname}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Save as .mp4
    out = cv2.VideoWriter(path, fourcc, fps, dims)
    # Write each frame to the video file
    for i in range(arr.shape[0]):
        out.write(arr[i])
    out.release()
