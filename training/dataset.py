import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=224):
        """
        root_dir: path to the real-vs-fake folder
        split: 'train', 'valid', or 'test'
        """
        self.samples = []

        split_dir = os.path.join(root_dir, split)

        for label, cls in enumerate(['real', 'fake']):
            cls_dir = os.path.join(split_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), label)
                    )

        print(f"[{split}] Loaded {len(self.samples)} images")

        self.transform = self._build_transform(split, img_size)

    def _build_transform(self, split, img_size):
        if split == 'train':
            return A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.ImageCompression(quality_range=(70, 100), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            result = self.transform(image=np.array(img))
            return result['image'], label
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a blank image if one fails, or handle differently
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            result = self.transform(image=blank)
            return result['image'], label