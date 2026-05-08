import os
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from dataset import DeepFakeDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ── Model ─────────────────────────────────────────────────────────────
class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        in_features = self.backbone._fc.in_features

        self.backbone._fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_feature_maps(self, x):
        return self.backbone.extract_features(x)


# ── Training loop ──────────────────────────────────────────────────────
def train(epochs=30, batch_size=16, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Use absolute paths relative to the script location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "real_vs_fake", "real-vs-fake")

    train_ds = DeepFakeDataset(DATA_DIR, split='train')
    val_ds   = DeepFakeDataset(DATA_DIR, split='valid')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=False)

    model = DeepFakeDetector().to(device)

    # Phase 1: freeze backbone, train head only (5 epochs)
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone._fc.parameters():
        param.requires_grad = True

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(5):
        _run_epoch(model, train_dl, optimizer, criterion, device, epoch, 'WARMUP')

    # Phase 2: unfreeze all, fine-tune with lower lr
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - 5)

    best_val_acc = 0.0
    for epoch in range(5, epochs):
        train_loss, train_acc = _run_epoch(
            model, train_dl, optimizer, criterion, device, epoch, 'TRAIN'
        )
        val_loss, val_acc = _run_epoch(
            model, val_dl, None, criterion, device, epoch, 'VAL'
        )
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(MODELS_DIR, 'efficientnet_b4_ff++.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f}) at {save_path}")


def _run_epoch(model, loader, optimizer, criterion, device, epoch, mode):
    is_train = (optimizer is not None)
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=f"[{mode}] Epoch {epoch:02d}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
            
            pbar.set_postfix(loss=total_loss/total, acc=correct/total)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"[{mode}] Epoch {epoch:02d} Finished | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    return avg_loss, acc


if __name__ == '__main__':
    train()