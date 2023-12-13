import copy
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import accuracy
from torchvision import transforms
from torchvision.models import EfficientNet_V2_S_Weights
from tqdm import tqdm

from motion_detection.dataset import CustomDataset
from motion_detection.models import build_efficientnet_v2_s
from motion_detection.utils import set_seed

set_seed(42)


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, writer):
    model.train()
    running_loss = 0
    total_size = 0
    pbar = tqdm(train_loader, total=len(train_loader), desc=f'Train', unit='batch')
    for batch in pbar:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        pred = model(images)
        loss = criterion(pred, labels)
        running_loss += loss.item()
        total_size += len(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix({'epoch': epoch, 'loss': running_loss / total_size})
        writer.add_scalar('Loss/train', loss.item(), epoch)

    writer.add_scalar('LR/train', scheduler.get_last_lr()[0], epoch)
    running_loss /= total_size
    return {'loss': running_loss}


@torch.no_grad()
def val_one_epoch(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    running_loss = 0
    total_size = 0
    all_labels = torch.tensor([], device=device)
    all_preds = torch.tensor([]).to(device, device=device)
    pbar = tqdm(val_loader, total=len(val_loader), desc=f'Val', unit='batch')
    for batch in pbar:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        pred = model(images)
        loss = criterion(pred, labels)
        all_labels = torch.cat((all_labels, labels))
        all_preds = torch.cat((all_preds, pred))
        running_loss += loss.item()
        total_size += len(images)
        pbar.set_postfix({'epoch': epoch, 'loss': running_loss / total_size, 'accuracy': 0})

    running_loss /= total_size
    acc = accuracy(all_preds, all_labels, 'binary', threshold=0.5)
    pbar.set_postfix({'epoch': epoch, 'loss': running_loss, 'accuracy': acc})
    writer.add_scalar('Loss/val', running_loss, epoch)
    writer.add_scalar('Accuracy/val', acc, epoch)
    return {'loss': running_loss, 'accuracy': acc}


def run_training(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, writer):
    start_time = time.time()
    best_accuracy = float('inf')
    best_epoch = -1

    save_dir = Path(f'checkpoints/{time.strftime("%Y%m%d-%H-%M-%S")}')
    save_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(epochs):
        _ = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, writer)
        val_history = val_one_epoch(model, val_loader, criterion, device, epoch, writer)

        if val_history['loss'] < best_accuracy:
            best_accuracy = val_history['loss']
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, f'{str(save_dir)}/best_model.pth')

    print(f'Best epoch: {best_epoch}, best accuracy: {best_accuracy}')
    print(f'Training time: {(time.time() - start_time) / 60:.2f} min')


def main():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=70, translate=(0.3, 0.3), scale=(0.5, 1.5)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms(),
    ])
    val_transform = transforms.Compose([
        EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
    ])
    train_dataset = CustomDataset(videos_dir='data/train',
                                  annotations_dir='data',
                                  transform=train_transform)
    val_dataset = CustomDataset(videos_dir='data/test',
                                annotations_dir='data',
                                transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = build_efficientnet_v2_s()

    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    writer = SummaryWriter()
    run_training(model, train_loader, val_loader, criterion, optimizer,
                 scheduler, device, epochs=50, writer=writer)


if __name__ == '__main__':
    main()
