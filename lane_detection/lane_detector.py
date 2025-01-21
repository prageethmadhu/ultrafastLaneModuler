import torch
import numpy as np
import scipy.special
from PIL import Image
from torchvision.transforms import transforms
from lane_detection.model.model import parsingNet
from lane_detection.constants import tusimple_row_anchor, culane_row_anchor
from lane_detection.transforms import get_transforms

class LaneDetector:
    def __init__(self, model_path, dataset="Tusimple", device="cpu"):
        self.device = torch.device(device)
        self.dataset = dataset

        # Dataset-specific configuration
        if dataset == "CULane":
            self.row_anchor = culane_row_anchor
            self.cls_num_per_lane = len(culane_row_anchor)  # Use dynamic length
        elif dataset == "Tusimple":
            self.row_anchor = tusimple_row_anchor
            self.cls_num_per_lane = 56  # Match checkpoint dimensions
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Initialize the model
        self.model = parsingNet(
            pretrained=False,
            backbone="18",  # Match the backbone in the checkpoint
            cls_dim=(100 + 1, self.cls_num_per_lane, 4),  # Match checkpoint: griding_num=100, num_lanes=4
            use_aux=False
        ).to(self.device)

        # Load pretrained weights
        state_dict = torch.load(model_path, map_location=self.device)["model"]
        self.model.load_state_dict({k[7:] if "module." in k else k: v for k, v in state_dict.items()}, strict=False)
        self.model.eval()

        # Image transformations
        self.transforms = get_transforms()

    def preprocess(self, frame):
        """
        Preprocess the input frame: Resize, Normalize, and Convert to Tensor.
        """
        img = Image.fromarray(frame)
        return self.transforms(img).unsqueeze(0).to(self.device)

    def detect_lanes(self, frame):
        """
        Detect lanes in the input frame.
        Args:
            frame (numpy.ndarray): Input frame (HxWxC).

        Returns:
            list: Detected lanes as lists of (x, y) coordinates.
        """
        # Preprocess the input frame
        img = self.preprocess(frame)

        # Perform inference
        with torch.no_grad():
            out = self.model(img)

        # Post-process the model output
        col_sample = np.linspace(0, 800 - 1, 100)  # Assume griding_num = 100
        col_sample_w = col_sample[1] - col_sample[0]

        out = out[0].data.cpu().numpy()
        out = out[:, ::-1, :]  # Reverse the order along the width
        prob = scipy.special.softmax(out[:-1, :, :], axis=0)
        idx = np.arange(100) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out = np.argmax(out, axis=0)
        loc[out == 100] = 0
        out = loc

        # Map output to lane points
        lanes = []
        for i in range(out.shape[1]):  # Iterate over columns (lanes)
            if np.sum(out[:, i] != 0) > 2:  # Check if valid points exist
                lane = []
                for k in range(min(len(self.row_anchor), self.cls_num_per_lane)):  # Avoid out-of-range index
                    if out[k, i] > 0:
                        x = int(out[k, i] * col_sample_w * frame.shape[1] / 800) - 1
                        y = int(frame.shape[0] * (self.row_anchor[len(self.row_anchor) - 1 - k] / 288)) - 1
                        lane.append((x, y))
                lanes.append(lane)

        return lanes
