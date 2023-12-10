import torch
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.transforms import Compose, Normalize

from src.motion_detection.dataset import CustomDataset

def collate_fn(batch):
    # defalt implementation
    return tuple(zip(*batch))

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    with torch.amp.autocast(device):
        for batch in train_loader:
            images, labels = batch
            print(images.shape, labels.shape)


def val_one_epoch(model, val_loader, criterion, device):
    pass


def main():
    transform = Compose([
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = CustomDataset(videos_dir='data/train',
                                  annotations_dir='data',
                                  transform=transform)
    val_dataset = CustomDataset(videos_dir='data/test',
                                annotations_dir='data',
                                transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = efficientnet_b0(weights=EfficientNet_B0_Weights)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    for epoch in range(1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_one_epoch(model, val_loader, criterion, device)


if __name__ == '__main__':
    main()
