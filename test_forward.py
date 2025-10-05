import torch
from model import VideoTokenMergingTransformer
import numpy as np

def test_forward():
    # Model parameters
    batch_size = 4
    num_frames = 60  # As per paper for LVU dataset
    num_channels = 3
    height = 224   # Input image size as per paper
    width = 224
    num_classes = 10

    # Create model
    model = VideoTokenMergingTransformer(
        num_classes=num_classes,
        num_frames=num_frames,
        patch_dim=1024,  # ViT-L dimension
        num_vtm_blocks=3,
        num_heads=8,
        dataset='LVU'  # Specify we're using ViT-L
    )
    model.cuda()
    model.eval()  # Set to eval mode to avoid auxiliary loss computation

    # Create dummy input
    # Shape: [batch_size, num_frames, channels, height, width]
    dummy_input = torch.randn(batch_size, num_frames, num_channels, height, width).cuda()

    # Forward pass
    print("Input shape:", dummy_input.shape)
    
    with torch.no_grad():
        outputs, aux_loss = model(dummy_input)
    
    print("Output shape:", outputs.shape)
    print("Output logits sample:", outputs[0][:5])  # Print first 5 logits of first batch item
    print("Auxiliary loss:", aux_loss)

    # Test if the output dimensions are correct
    assert outputs.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {outputs.shape}"
    print("\nForward pass successful! ✅")

if __name__ == "__main__":
    video_features = np.load('/data_local3/susimmuk/lvu/P17-stereo-P17_milk_ch0.npy')
    print(video_features.shape)
    # test_forward()