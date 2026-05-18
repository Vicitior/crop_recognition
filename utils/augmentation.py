"""
数据增强模块
提供训练和验证阶段的图像预处理管道

新增功能：
- 类别级自适应增强（针对难分类别）
- 基于遗传算法的增强参数优化
- 支持Mixup、CutMix、RandomErasing等高级增强
"""
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
import torch

# ImageNet标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# 难分类别配置（根据训练结果调整）
DIFFICULT_CLASSES = {
    "cotton_boll_setting": {
        "difficulty": 0.9,  # 难度系数（0-1）
        "augmentation_strength": 1.5,  # 增强强度倍数
        "strategies": ["mixup", "cutout", "color_jitter", "rotation"]
    },
    "cotton_flowering": {
        "difficulty": 0.7,
        "augmentation_strength": 1.2,
        "strategies": ["mixup", "color_jitter"]
    }
}


def get_train_transforms(img_size=224, augmentation_level="medium"):
    """
    获取训练时的数据增强

    Args:
        img_size: 图像尺寸
        augmentation_level: 增强级别 (basic/medium/strong)
    """
    if augmentation_level == "basic":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    elif augmentation_level == "medium":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    elif augmentation_level == "strong":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
        ])
    else:
        raise ValueError(f"未知的增强级别: {augmentation_level}")


def get_class_specific_transforms(class_name, img_size=224):
    """
    获取类别特定的数据增强

    Args:
        class_name: 类别名称
        img_size: 图像尺寸
    """
    config = DIFFICULT_CLASSES.get(class_name, None)

    if config is None:
        # 非难分类别，使用标准增强
        return get_train_transforms(img_size, "medium")

    difficulty = config["difficulty"]
    strength = config["augmentation_strength"]

    # 根据难度调整增强参数
    scale_range = (max(0.5, 0.7 - difficulty * 0.2), 1.0)
    rotation_range = int(30 + difficulty * 30)
    color_strength = 0.3 + difficulty * 0.2

    transform_list = [
        transforms.RandomResizedCrop(img_size, scale=scale_range),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(rotation_range),
        transforms.ColorJitter(
            brightness=color_strength,
            contrast=color_strength,
            saturation=color_strength,
            hue=0.1 + difficulty * 0.1
        ),
    ]

    # 根据策略添加特定增强
    strategies = config["strategies"]

    if "cutout" in strategies:
        transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)))

    if "color_jitter" in strategies:
        transform_list.append(transforms.RandomGrayscale(p=0.15))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    return transforms.Compose(transform_list)


def get_val_transforms(img_size=224):
    """获取验证时的数据增强"""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


def get_predict_transforms(img_size=224):
    """获取预测时的数据增强"""
    return get_val_transforms(img_size)


# ============================================================
# Mixup / CutMix 实现
# ============================================================

def mixup_data(x, y, alpha=0.4):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # 生成随机框
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    # 混合图像
    x_clone = x.clone()
    x_clone[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # 调整lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]
    return x_clone, y_a, y_b, lam


def rand_bbox(size, lam):
    """生成随机边界框"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup/CutMix损失计算"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# 自适应增强调度器
# ============================================================

class AdaptiveAugmentationScheduler:
    """
    自适应增强调度器
    根据训练进度和类别难度动态调整增强强度
    """

    def __init__(self, total_epochs, class_names, warmup_epochs=5):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.class_names = class_names

        # 初始化类别难度
        self.class_difficulty = {name: 0.5 for name in class_names}

        # 增强强度历史
        self.augmentation_history = []

    def update_difficulty(self, class_accuracies):
        """
        根据类别准确率更新难度

        Args:
            class_accuracies: dict, 类别名 -> 准确率
        """
        for cls_name, acc in class_accuracies.items():
            if cls_name in self.class_difficulty:
                # 准确率越低，难度越高
                self.class_difficulty[cls_name] = 1.0 - (acc / 100.0)

    def get_augmentation_params(self, epoch):
        """
        获取当前epoch的增强参数

        Args:
            epoch: 当前epoch

        Returns:
            dict: 增强参数
        """
        # 训练进度
        progress = epoch / self.total_epochs

        # 基础强度（从弱到强再到弱）
        if epoch < self.warmup_epochs:
            base_strength = 0.3 + 0.7 * (epoch / self.warmup_epochs)
        else:
            # 余弦退火
            progress_after_warmup = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            base_strength = 0.5 + 0.5 * np.cos(np.pi * progress_after_warmup)

        params = {
            "mixup_alpha": 0.4 * base_strength,
            "cutmix_alpha": 1.0 * base_strength,
            "mixup_prob": 0.5 * base_strength,
            "cutmix_prob": 0.3 * base_strength,
            "class_difficulty": self.class_difficulty.copy()
        }

        self.augmentation_history.append(params)
        return params


# ============================================================
# 遗传算法优化增强参数
# ============================================================

class AugmentationGeneticOptimizer:
    """
    使用遗传算法优化数据增强参数

    参考论文: arXiv:2406.13081
    """

    def __init__(self, population_size=20, generations=10):
        self.population_size = population_size
        self.generations = generations

        # 增强参数范围
        self.param_ranges = {
            "rotation_range": (0, 60),
            "brightness_range": (0.1, 0.5),
            "contrast_range": (0.1, 0.5),
            "saturation_range": (0.1, 0.5),
            "hue_range": (0.0, 0.2),
            "scale_min": (0.5, 0.9),
            "erase_prob": (0.0, 0.5),
            "mixup_alpha": (0.0, 1.0),
        }

    def _initialize_population(self):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param, (low, high) in self.param_ranges.items():
                individual[param] = np.random.uniform(low, high)
            population.append(individual)
        return population

    def _crossover(self, parent1, parent2):
        """交叉操作"""
        child = {}
        for param in self.param_ranges:
            if np.random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child

    def _mutate(self, individual, mutation_rate=0.1):
        """变异操作"""
        mutated = individual.copy()
        for param, (low, high) in self.param_ranges.items():
            if np.random.random() < mutation_rate:
                mutated[param] = np.random.uniform(low, high)
        return mutated

    def optimize(self, fitness_func, verbose=True):
        """
        运行遗传算法优化

        Args:
            fitness_func: 适应度函数，接受参数字典，返回适应度分数
            verbose: 是否打印进度

        Returns:
            最优参数和适应度分数
        """
        population = self._initialize_population()
        best_individual = None
        best_fitness = -float('inf')

        for generation in range(self.generations):
            # 计算适应度
            fitness_scores = []
            for individual in population:
                fitness = fitness_func(individual)
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()

            if verbose:
                print(f"Generation {generation + 1}/{self.generations}, "
                      f"Best Fitness: {best_fitness:.4f}")

            # 选择
            fitness_scores = np.array(fitness_scores)
            fitness_scores = fitness_scores - fitness_scores.min() + 1e-6
            probabilities = fitness_scores / fitness_scores.sum()

            # 生成下一代
            new_population = []

            # 精英保留
            elite_idx = np.argmax(fitness_scores)
            new_population.append(population[elite_idx])

            while len(new_population) < self.population_size:
                # 选择父代
                parent1_idx = np.random.choice(len(population), p=probabilities)
                parent2_idx = np.random.choice(len(population), p=probabilities)

                # 交叉
                child = self._crossover(population[parent1_idx], population[parent2_idx])

                # 变异
                child = self._mutate(child)

                new_population.append(child)

            population = new_population

        return best_individual, best_fitness


# ============================================================
# 工厂函数
# ============================================================

def create_augmentation_pipeline(class_name=None, img_size=224, augmentation_level="medium"):
    """
    创建数据增强管道

    Args:
        class_name: 类别名称（可选，用于类别特定增强）
        img_size: 图像尺寸
        augmentation_level: 增强级别

    Returns:
        transforms.Compose: 数据增强管道
    """
    if class_name and class_name in DIFFICULT_CLASSES:
        return get_class_specific_transforms(class_name, img_size)
    else:
        return get_train_transforms(img_size, augmentation_level)
