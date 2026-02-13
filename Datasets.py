import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import random
import numpy as np
import os
import cv2

# ============================================================================
# MASK GENERATOR
# ============================================================================
class MaskGenerator:
    @staticmethod
    def generate_hsv_disease_mask(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        lower_brown = np.array([10, 30, 20])
        upper_brown = np.array([30, 180, 150])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        lower_yellow = np.array([20, 40, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        disease_mask = cv2.bitwise_or(brown_mask, yellow_mask)
        disease_mask = cv2.bitwise_or(disease_mask, dark_mask)
        
        return disease_mask
    
    @staticmethod
    def generate_lab_disease_mask(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        _, a_thresh = cv2.threshold(a_channel, 135, 255, cv2.THRESH_BINARY)
        _, b_thresh = cv2.threshold(b_channel, 135, 255, cv2.THRESH_BINARY)
        
        lab_mask = cv2.bitwise_or(a_thresh, b_thresh)
        
        return lab_mask
    
    @staticmethod
    def generate_edge_disease_mask(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        edges = cv2.Canny(bilateral, 30, 100)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_mask = np.zeros_like(gray)
        cv2.drawContours(edge_mask, contours, -1, 255, thickness=cv2.FILLED)
        
        return edge_mask
    
    @staticmethod
    def generate_texture_disease_mask(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        gabor_responses = []
        for theta in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel((21, 21), 5, np.deg2rad(theta), 10, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            gabor_responses.append(filtered)
        
        gabor_combined = np.max(np.stack(gabor_responses), axis=0).astype(np.uint8)
        
        _, texture_mask = cv2.threshold(gabor_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return texture_mask
    
    @staticmethod
    def generate_statistical_mask(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        masks = []
        for channel in cv2.split(lab):
            local_mean = uniform_filter(channel.astype(float), size=15)
            local_var = uniform_filter(channel.astype(float)**2, size=15) - local_mean**2
            local_var = np.maximum(local_var, 0)
            
            var_range = local_var.max() - local_var.min()
            if var_range < 1e-8:
                var_norm = np.zeros_like(local_var, dtype=np.uint8)
            else:
                var_norm = ((local_var - local_var.min()) / (var_range + 1e-8) * 255).astype(np.uint8)
            _, var_mask = cv2.threshold(var_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masks.append(var_mask)
        
        stat_mask = cv2.bitwise_or(masks[0], masks[1])
        stat_mask = cv2.bitwise_or(stat_mask, masks[2])
        
        return stat_mask
    
    @staticmethod
    def ensemble_mask_generation(image: np.ndarray, methods: Optional[List[str]] = None) -> np.ndarray:
        if methods is None:
            methods = ['hsv', 'lab', 'edge', 'texture', 'statistical']
        
        masks = []
        
        if 'hsv' in methods:
            masks.append(MaskGenerator.generate_hsv_disease_mask(image))
        if 'lab' in methods:
            masks.append(MaskGenerator.generate_lab_disease_mask(image))
        if 'edge' in methods:
            masks.append(MaskGenerator.generate_edge_disease_mask(image))
        if 'texture' in methods:
            masks.append(MaskGenerator.generate_texture_disease_mask(image))
        if 'statistical' in methods:
            masks.append(MaskGenerator.generate_statistical_mask(image))
        
        mask_stack = np.stack([m / 255.0 for m in masks], axis=0)
        vote_mask = (mask_stack.mean(axis=0) > 0.5).astype(np.uint8) * 255
        
        vote_mask = MaskGenerator.post_process_mask(vote_mask, image)
        
        return vote_mask
    
    @staticmethod
    def post_process_mask(mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_size = int((mask.shape[0] * mask.shape[1]) * 0.001)
        
        filtered_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                filtered_mask[labels == i] = 255
        
        filtered_mask = cv2.GaussianBlur(filtered_mask, (5, 5), 0)
        _, filtered_mask = cv2.threshold(filtered_mask, 127, 255, cv2.THRESH_BINARY)
        
        return filtered_mask

# ============================================================================
# CUSTOM COLLATE FUNCTION
# ============================================================================
def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    
    mask_shapes = [item[1].shape for item in batch]
    all_same_shape = all(s == mask_shapes[0] for s in mask_shapes)
    
    if all_same_shape:
        masks = torch.stack([item[1] for item in batch])
    else:
        masks = [item[1] for item in batch]
    
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    
    original_sizes = [item[3] for item in batch]
    
    return images, masks, labels, original_sizes

# ============================================================================
# DATASET WITH IMPROVED MASK GENERATION
# ============================================================================
class CottonDiseaseDataset(Dataset):
    def __init__(self, config, split: str = 'train'):
        self.config = config
        self.split = split
        self.data_root = Path(config.data_root)
        self.samples = []
        self.classes = sorted([d.name for d in self.data_root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.mask_generator = MaskGenerator()
        print(f"Classes found: {len(self.classes)}")
        print(f"Classes: {self.classes}")
        
        for class_name in self.classes:
            class_dir = self.data_root / class_name
            for img_path in sorted(list(class_dir.glob('*.*')) + list(class_dir.glob('*.png'))):
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"Total samples: {len(self.samples)}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Tuple[int, int]]:
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        image_np = np.array(image)
        mask = self.generate_mask(image_np)
        
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')
        
        if self.split == 'train' and getattr(self.config, 'augment', False):
            image_transformed, mask_transformed = self.apply_augmentations(image, mask_pil)
        else:
            image_transformed = transforms.Resize((self.config.image_size, self.config.image_size))(image)
            if self.split in ['test', 'val'] and getattr(self.config, 'preserve_original_size_eval', True):
                mask_transformed = mask_pil
            else:
                mask_transformed = transforms.Resize(
                    (self.config.image_size, self.config.image_size), 
                    interpolation=transforms.InterpolationMode.NEAREST
                )(mask_pil)
        
        image_tensor = transforms.ToTensor()(image_transformed)
        mask_tensor = transforms.ToTensor()(mask_transformed)
        
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image_tensor = normalize(image_tensor)
        
        return image_tensor, mask_tensor, label, original_size

    def apply_augmentations(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        if random.random() < 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        
        if random.random() < 0.5:
            angle = random.uniform(
                -getattr(self.config, 'rotation_degrees', 30), 
                getattr(self.config, 'rotation_degrees', 30)
            )
            image = transforms.functional.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = transforms.functional.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
        
        image = transforms.Resize((self.config.image_size, self.config.image_size))(image)
        mask = transforms.Resize(
            (self.config.image_size, self.config.image_size), 
            interpolation=transforms.InterpolationMode.NEAREST
        )(mask)
        
        if random.random() < getattr(self.config, 'augment_prob', 0.5) and getattr(self.config, 'use_advanced_augmentation', False):
            color_jitter = transforms.ColorJitter(
                brightness=getattr(self.config, 'color_jitter_brightness', 0.2),
                contrast=getattr(self.config, 'color_jitter_contrast', 0.2),
                saturation=getattr(self.config, 'color_jitter_saturation', 0.2),
                hue=getattr(self.config, 'color_jitter_hue', 0.1)
            )
            image = color_jitter(image)
        
        return image, mask

    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        method = getattr(self.config, 'mask_method', 'ensemble')
        
        if method == 'ensemble':
            mask = self.mask_generator.ensemble_mask_generation(image)
        elif method == 'hsv':
            mask = self.mask_generator.generate_hsv_disease_mask(image)
        elif method == 'lab':
            mask = self.mask_generator.generate_lab_disease_mask(image)
        elif method == 'edge':
            mask = self.mask_generator.generate_edge_disease_mask(image)
        elif method == 'texture':
            mask = self.mask_generator.generate_texture_disease_mask(image)
        elif method == 'statistical':
            mask = self.mask_generator.generate_statistical_mask(image)
        else:
            mask = self.mask_generator.ensemble_mask_generation(image)
        
        if method != 'ensemble':
            mask = self.mask_generator.post_process_mask(mask, image)
            
        return mask.astype(np.float32) / 255.0

    def __len__(self) -> int:
        return len(self.samples)

def create_dataloaders(config: CAMDFNetConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    from torch.utils.data import random_split
    full_dataset = CottonDiseaseDataset(config.dataset, split=config.training.split)
    
    # Set num_classes automatically from dataset
    config.model.num_classes = len(full_dataset.classes)
    print(f"\n📊 Dataset Information:")
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Number of classes: {config.model.num_classes}")
    print(f"  Classes: {full_dataset.classes}")
    
    total_size = len(full_dataset)
    train_size = int(total_size * config.dataset.train_ratio)
    val_size = int(total_size * config.dataset.val_ratio)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(config.system.seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader

# ============================================================================
# AUGMENTATION & LOSS FUNCTIONS
# ============================================================================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam