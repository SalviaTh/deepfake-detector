import torch, os
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader
from dataset import DeepFakeDataset
from train import DeepFakeDetector

def evaluate(weights_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if weights_path is None:
        weights_path = os.path.join(SCRIPT_DIR, "..", "models", "efficientnet_b4_ff++.pth")

    model = DeepFakeDetector()
    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found at {weights_path}")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()

    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "real_vs_fake", "real-vs-fake")

    test_ds = DeepFakeDataset(DATA_DIR, split='test')
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, 1)[:, 1]
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    print(classification_report(all_labels, all_preds,
                                 target_names=['Real', 'Fake']))
    auc = roc_auc_score(all_labels, all_probs)
    print(f"ROC-AUC: {auc:.4f}")


if __name__ == '__main__':
    evaluate()