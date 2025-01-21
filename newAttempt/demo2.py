import torch, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image
import time


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print("Start testing...")
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    # Dataset configuration
    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError("Dataset not supported!")

    # Initialize model
    net = parsingNet(
        pretrained=False, # not get weight from imagenet
        backbone=cfg.backbone,
        cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
        use_aux=False,
    ).to(torch.device("cpu"))

    state_dict = torch.load(cfg.test_model, map_location="cpu")["model"]
    net.load_state_dict({k[7:] if "module." in k else k: v for k, v in state_dict.items()}, strict=False)
    net.eval()

    # Image transformations
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Video input and output
    video_input_path = "crash_ori3.mp4"
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
        print(f"Processing frame {frame_count + 1}")
        if not ret:
            break

        frame_count += 1
        start_time = time.time()
        # Convert frame to PIL Image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Apply transformations
        img = img_transforms(img).unsqueeze(0).to(torch.device("cpu"))

        with torch.no_grad():
            out_net = net(img)
        print(f"Inference time: {time.time() - start_time:.2f} seconds")
        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_net = out_net[0].data.cpu().numpy()
        out_net = out_net[:, ::-1, :]
        prob = scipy.special.softmax(out_net[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_net = np.argmax(out_net, axis=0)
        loc[out_net == cfg.griding_num] = 0
        out_net = loc

        # Annotate frame
        for i in range(out_net.shape[1]):
            if np.sum(out_net[:, i] != 0) > 2:
                for k in range(min(cls_num_per_lane, len(row_anchor))):
                    if out_net[k, i] > 0:
                        ppp = (
                            int(out_net[k, i] * col_sample_w * width / 800) - 1,
                            int(height * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1,
                        )
                        cv2.circle(frame, ppp, 5, (0, 255, 0), -1)
        print("Writing frame to output video...")
        # Write to output video
        out.write(frame)

    print(f"Total frames processed: {frame_count}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved to {video_output_path}")
