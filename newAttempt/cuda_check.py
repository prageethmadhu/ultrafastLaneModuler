import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image
import time

cap = cv2.VideoCapture(0)  # Use webcam
out = cv2.VideoWriter("sample.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
for _ in range(50):  # Capture 50 frames
    ret, frame = cap.read()
    if ret:
        out.write(frame)
cap.release()
out.release()
