from torchvision import transforms

def get_transforms():
    return transforms.Compose([
        transforms.Resize((288, 800)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Standard ImageNet normalization
    ])
