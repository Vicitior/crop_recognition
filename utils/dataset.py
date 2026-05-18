"""
数据集加载模块
支持从标准目录结构加载农作物图像数据集
"""
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from utils.augmentation import get_train_transforms, get_val_transforms


def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transforms())
    val_dataset = datasets.ImageFolder(val_dir, transform=get_val_transforms())

    # 处理类别不平衡：使用加权采样
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    weights = [1.0 / class_counts[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    class_names = train_dataset.classes
    return train_loader, val_loader, class_names


def get_test_dataloader(data_dir, batch_size=16, num_workers=4):
    test_dir = os.path.join(data_dir, "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=get_val_transforms())
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return test_loader, test_dataset.classes
