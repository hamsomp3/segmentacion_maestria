# https://www.youtube.com/watch?v=IHq1t7NxS8k&ab_channel=AladdinPersson

import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from unet_model import UNET


from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

num_classes = 6
# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "img/02 - Denoised/train/ims/"
TRAIN_MASK_DIR = "img/02 - Denoised/train/masks/"
VAL_IMG_DIR = "img/02 - Denoised/test/ims/"
VAL_MASK_DIR = "img/02 - Denoised/test/mask/"




def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for data, targets in loop:
        data, targets = data.to(DEVICE), targets.to(DEVICE, dtype=torch.long)
        #targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0],  # Un solo valor para un canal
            std=[1.0],   # Un solo valor para un canal
            max_pixel_value=1.0,
        ),
        ToTensorV2(),
        ],
    )
    val_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0],  # Un solo valor para un canal
            std=[1.0],   # Un solo valor para un canal
            max_pixel_value=1.0,
        ),
        ToTensorV2(),
        ],
    )

    model = UNET(in_channels=1, out_channels=num_classes).to(DEVICE) 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        check_accuracy(val_loader, model, device=DEVICE)
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed. Checkpoint saved.")

if __name__ == "__main__":
    main()