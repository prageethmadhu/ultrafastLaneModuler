import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# ParsingNet model
class parsingNet(torch.nn.Module):
    def __init__(self, pretrained=False, backbone="18", cls_dim=(101, 56, 4), use_aux=False):
        super(parsingNet, self).__init__()
        # Updated dimensions to match the checkpoint
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(800, 2048),  # Align input feature dimensions
            torch.nn.ReLU(),
            torch.nn.Linear(2048, cls_dim[0] * cls_dim[1] * cls_dim[2])  # Align output with checkpoint
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        return self.cls(x)

if __name__ == "__main__":
    # Configuration
    cfg = {
        "backbone": "18",
        "dataset": "Tusimple",
        "test_model": "pretrained_weights/tusimple_18.pth",
        "griding_num": 100,
    }

    print("Start testing...")

    # Initialize model
    net = parsingNet(
        pretrained=False,
        backbone=cfg["backbone"],
        cls_dim=(cfg["griding_num"] + 1, 56, 4),
    ).to(torch.device("cpu"))

    # Load weights
    try:
        state_dict = torch.load(cfg["test_model"], map_location="cpu")["model"]
        # Filter state_dict to avoid dimension mismatches
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k in net.state_dict() and net.state_dict()[k].shape == v.shape
        }
        missing_keys, unexpected_keys = net.load_state_dict(filtered_state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        net.eval()
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit(1)

    # Image transformations
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),  # Match training resolution
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Video input and output
    video_input_path = "crash_ori.mp4"
    video_output_path = "crash2.mp4"

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        raise ValueError("Error: Unable to open the input video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Input Video Dimensions: {width}x{height}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read or error in reading frames.")
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Convert frame to PIL Image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Apply transformations
        img = img_transforms(img).unsqueeze(0).to(torch.device("cpu"))

        # Model inference
        try:
            with torch.no_grad():
                out_net = net(img)
                print(f"Model output shape: {out_net.shape}")
        except Exception as e:
            print(f"Error during inference: {e}")
            continue

        print(f"Frame {frame_count} processed.")
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved to {video_output_path}")
