import torch
import torchvision
from dataset_unet import organ_segmentation_dataset
from torch.utils.data import DataLoader
num_classes = 6

def save_checkpoint(state, filename="my_checkpoint.pth.tar"): #revisar esta parte porque ahora se tiene segnet
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):
    train_ds = organ_segmentation_dataset(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_ds = organ_segmentation_dataset(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    class_correct = [0. for _ in range(num_classes)]
    class_total = [0. for _ in range(num_classes)]

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.softmax(model(x), dim=1)
            preds = preds.argmax(dim=1)
            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(y)

            # Calculate Dice Score and IoU per batch
            for cls in range(num_classes):
                pred_mask = (preds == cls)
                true_mask = (y == cls)
                intersection = (pred_mask & true_mask).sum()
                union = (pred_mask | true_mask).sum()
                dice_score += (2. * intersection + 1e-8) / (pred_mask.sum() + true_mask.sum() + 1e-8)
                iou_score += (intersection + 1e-8) / (union + 1e-8)
                class_correct[cls] += intersection.item()
                class_total[cls] += true_mask.sum().item()

    # Calculate total Dice and IoU
    total_classes = len(class_correct)
    overall_dice = dice_score / (len(loader) * total_classes)
    overall_iou = iou_score / (len(loader) * total_classes)

    # Print accuracy stats
    print(f"Total correct: {num_correct}/{num_pixels} with pixel accuracy: {num_correct/num_pixels*100:.2f}%")
    print(f"Overall Dice score: {overall_dice}")
    print(f"Overall IoU score: {overall_iou}")
    for i in range(total_classes):
        if class_total[i]:
            print(f"Class {i} - Accuracy: {class_correct[i] / class_total[i] * 100:.2f}%")

    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.softmax(model(x), dim=1)
            preds = preds.argmax(dim=1, keepdim=True)
        
        # Convierte preds a uint8 antes de guardarlas
        preds = preds.to(dtype=torch.uint8)  # Asegúrate que los valores están en el rango correcto
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        
        # También debes asegurarte que y está en uint8 si quieres guardar la máscara original
        y_uint8 = y.to(dtype=torch.uint8).unsqueeze(1)  # Añade un canal para que coincida con preds
        torchvision.utils.save_image(y_uint8, f"{folder}/{idx}.png")
    
    model.train()