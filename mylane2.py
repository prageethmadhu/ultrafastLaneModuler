import logging
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import scipy.special
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define ParsingNet architecture
class ParsingNet(torch.nn.Module):
    def __init__(self, cls_dim=(101, 56, 4), backbone="18"):
        super(ParsingNet, self).__init__()
        self.cls_dim = cls_dim  # griding_num + 1, row anchors, num lanes
        # Simplified backbone based on ResNet-18
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((36, 101)),
        )
        self.fc = torch.nn.Linear(128 * 36 * 101, np.prod(cls_dim))

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1, *self.cls_dim)

# Minimal configuration
griding_num = 100  # Number of discretized points across the width of the image
cls_num_per_lane = 56  # Number of row anchors (TuSimple dataset)
row_anchor = np.linspace(160, 710, cls_num_per_lane).tolist()  # TuSimple row anchors
device = torch.device("cpu")
video_input_path = "input_video.mp4"
video_output_path = "output_video.mp4"

logger.info("Initializing ParsingNet model...")
# Initialize model
model = ParsingNet(cls_dim=(griding_num + 1, cls_num_per_lane, 4)).to(device)

# Load weights from tusimple_18.pth
try:
    logger.info("Loading model weights from 'tusimple_18.pth'...")
    state_dict = torch.load("tusimple_18.pth", map_location=device)["model"]
    model.load_state_dict({k[7:] if "module." in k else k: v for k, v in state_dict.items()}, strict=False)
    logger.info("Model weights loaded successfully.")
except FileNotFoundError:
    logger.error("Model weights file 'tusimple_18.pth' not found. Please check the path.")
    exit(1)

model.eval()
logger.info("Model initialized and set to evaluation mode.")

# Image transformations
img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Process video
logger.info(f"Opening video file '{video_input_path}'...")
cap = cv2.VideoCapture(video_input_path)
if not cap.isOpened():
    logger.error(f"Failed to open video file '{video_input_path}'. Exiting.")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
logger.info(f"Video dimensions: {width}x{height}, FPS: {fps}, Total frames: {frame_total}")

out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logger.info("End of video or error reading frame.")
        break

    frame_count += 1
    progress = (frame_count / frame_total) * 100
    logger.info(f"Processing frame {frame_count}/{frame_total} ({progress:.2f}% complete)...")

    # Preprocess frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img_transforms(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img)

    # Postprocess and visualize lanes
    output = output[0].cpu().numpy()
    prob = scipy.special.softmax(output[:-1, :, :], axis=0)
    loc = np.sum(prob * np.arange(griding_num).reshape(-1, 1, 1), axis=0)

    for lane_idx in range(loc.shape[1]):
        for anchor_idx in range(len(row_anchor)):
            if loc[anchor_idx, lane_idx] > 0:
                point = (
                    int(loc[anchor_idx, lane_idx] * width / griding_num),
                    int(row_anchor[anchor_idx] * height / 720),
                )
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

    out.write(frame)

logger.info(f"Video processing complete. Total frames processed: {frame_count}")
logger.info(f"Output video saved to '{video_output_path}'")

cap.release()
out.release()
cv2.destroyAllWindows()

elapsed_time = time.time() - start_time
logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
logger.info("All done!")
