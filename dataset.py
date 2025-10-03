import torch
from torch.utils.data import Dataset, DataLoader

class DummyVideoDataset(Dataset):
    """
    A dummy dataset that generates random video tensors of shape (T, C, H, W).
    Replace this with the actual dataset loading logic for LVU, COIN, or Breakfast.
    """
    def __init__(self, num_samples=100, num_frames=60, channels=3, height=224, width=224, num_classes=10):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.video_dims = (channels, height, width)
        self.num_classes = num_classes
        print(f"Dummy dataset created with {num_samples} samples.")
        print(f"Video specs: {num_frames} frames, {height}x{width} resolution.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random video tensor and a random label
        # Shape: (T, C, H, W)
        video = torch.randn(self.num_frames, *self.video_dims)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return video, label

def get_dataloader(batch_size=4, num_frames=60, num_workers=0):
    """Creates and returns a DataLoader for the dummy video dataset."""
    dataset = DummyVideoDataset(num_samples=batch_size * 10, num_frames=num_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
