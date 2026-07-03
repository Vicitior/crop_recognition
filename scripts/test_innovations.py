"""
创新模块单元测试

验证三个创新模块的 forward 输出 shape 正确性
"""

import sys
import os
from pathlib import Path
import torch

# 修复 Windows 控制台编码问题
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_phenology_graph():
    """测试生育期图建模模块"""
    from models.phenology_graph import (
        build_gaussian_adjacency,
        GraphConvStageHead,
        PhenologyAwareLoss,
        PhenologyAwareClassifier,
    )

    print("=" * 50)
    print("测试 生育期图建模 (Phenology Graph)")
    print("=" * 50)

    B, D = 4, 768  # batch_size=4, feat_dim=768

    # 1. 测试邻接矩阵
    adj = build_gaussian_adjacency(5, sigma=1.5)
    assert adj.shape == (5, 5), f"邻接矩阵 shape 错误: {adj.shape}"
    print(f"✓ 邻接矩阵: {adj.shape}")
    print(f"  对角线值: {[f'{adj[i][i]:.4f}' for i in range(5)]}")
    print(f"  相邻值:   {[f'{adj[i][i+1]:.4f}' for i in range(4)]}")

    # 2. 测试 GraphConvStageHead
    head = GraphConvStageHead(feat_dim=D, hidden_dim=256, num_stages=5)
    features = torch.randn(B, D)
    stage_logits = head(features)
    assert stage_logits.shape == (B, 5), \
        f"StageHead 输出 shape 错误: {stage_logits.shape}"
    print(f"✓ GraphConvStageHead: input({B}, {D}) → output{stage_logits.shape}")

    # 3. 测试 PhenologyAwareLoss
    loss_fn = PhenologyAwareLoss(num_stages=5, sigma=1.5, alpha=0.5, beta=0.1)
    labels = torch.randint(0, 5, (B,))
    loss = loss_fn(stage_logits, labels)
    assert loss.dim() == 0, f"损失应该是标量: {loss.shape}"
    print(f"✓ PhenologyAwareLoss: loss={loss.item():.4f}")

    # 4. 测试完整 PhenologyAwareClassifier
    classifier = PhenologyAwareClassifier(
        feat_dim=D, num_classes=15, hidden_dim=256
    )
    features = torch.randn(B, D)
    logits, crop_logits = classifier(features, threshold=0.7)
    assert logits.shape == (B, 15), \
        f"全局 logits shape 错误: {logits.shape}"
    assert crop_logits.shape == (B, 3), \
        f"作物 logits shape 错误: {crop_logits.shape}"
    print(f"✓ PhenologyAwareClassifier: "
          f"logits{logits.shape}, crop_logits{crop_logits.shape}")

    # 5. 测试低置信度路由（threshold=1.0 时应全部走软路由）
    logits_soft, _ = classifier(features, threshold=1.0)
    print(f"✓ 软路由测试: logits{logits_soft.shape}")

    print("✓ 生育期图建模 全部测试通过!\n")


def test_confidence_router():
    """测试置信度路由模块"""
    from models.confidence_router import (
        CropRouter,
        StageBranch,
        ConfidenceRouterClassifier,
        ConfidenceRouterLoss,
    )

    print("=" * 50)
    print("测试 置信度路由 (Confidence Router)")
    print("=" * 50)

    B, D = 4, 768

    # 1. 测试 CropRouter
    router = CropRouter(feat_dim=D, hidden_dim=256, num_crops=3)
    features = torch.randn(B, D)
    crop_logits = router(features)
    assert crop_logits.shape == (B, 3), \
        f"CropRouter 输出 shape 错误: {crop_logits.shape}"
    print(f"✓ CropRouter: input({B}, {D}) → output{crop_logits.shape}")

    # 2. 测试 StageBranch
    branch = StageBranch(feat_dim=D, num_stages=5, hidden_dim=256)
    stage_logits = branch(features)
    assert stage_logits.shape == (B, 5), \
        f"StageBranch 输出 shape 错误: {stage_logits.shape}"
    print(f"✓ StageBranch: input({B}, {D}) → output{stage_logits.shape}")

    # 3. 测试完整 ConfidenceRouterClassifier
    classifier = ConfidenceRouterClassifier(
        feat_dim=D, num_classes=15, hidden_dim=256
    )

    # 测试硬路由（threshold=0.7）
    logits, crop_logits = classifier(features, threshold=0.7)
    assert logits.shape == (B, 15), \
        f"硬路由 logits shape 错误: {logits.shape}"
    print(f"✓ 硬路由(threshold=0.7): logits{logits.shape}")

    # 测试软路由（threshold=0.0，所有分支加权）
    logits_soft, _ = classifier(features, threshold=0.0)
    assert logits_soft.shape == (B, 15)
    print(f"✓ 软路由(threshold=0.0): logits{logits_soft.shape}")

    # 测试返回路由信息
    logits, crop_logits, info = classifier(
        features, threshold=0.7, return_routing_info=True
    )
    print(f"✓ 路由信息: 硬路由{info['hard_count']}个, "
          f"软路由{info['soft_count']}个, "
          f"硬路由比例{info['hard_ratio']:.2%}")

    # 4. 测试损失函数
    loss_fn = ConfidenceRouterLoss(lambda_crop=0.3, gamma_entropy=0.05)
    stage_labels = torch.randint(0, 15, (B,))
    crop_labels = torch.randint(0, 3, (B,))
    loss, details = loss_fn(logits, crop_logits, stage_labels, crop_labels)
    assert loss.dim() == 0
    print(f"✓ ConfidenceRouterLoss: loss={loss.item():.4f}")
    print(f"  详情: {details}")

    print("✓ 置信度路由 全部测试通过!\n")


def test_adaptive_lora():
    """测试 Adaptive LoRA 模块"""
    from models.adaptive_lora import (
        AdaptiveLoRAExpert,
        AdaptiveLoRALinear,
    )

    print("=" * 50)
    print("测试 Adaptive LoRA")
    print("=" * 50)

    B, D_in, D_out = 4, 768, 768

    # 1. 测试不同 rank 的 LoRAExpert
    for rank in [4, 8, 16]:
        expert = AdaptiveLoRAExpert(D_in, D_out, rank=rank, alpha=rank*2)
        x = torch.randn(B, D_in)
        out = expert(x)
        assert out.shape == (B, D_out), \
            f"LoRAExpert(rank={rank}) shape 错误: {out.shape}"
        params = sum(p.numel() for p in expert.parameters())
        print(f"✓ LoRAExpert(rank={rank}): {out.shape}, {params:,} 参数")

    # 2. 测试 AdaptiveLoRALinear
    original_linear = torch.nn.Linear(D_in, D_out)
    crop_ranks = {0: 4, 1: 8, 2: 16}

    adaptive_lora = AdaptiveLoRALinear(
        original_linear, crop_ranks, alpha_multiplier=2.0
    )

    x = torch.randn(B, D_in)

    # 测试指定 crop_id
    for crop_id in [0, 1, 2]:
        crop_ids = torch.tensor([crop_id] * B)
        out = adaptive_lora(x, crop_ids)
        assert out.shape == (B, D_out), \
            f"AdaptiveLoRALinear(crop={crop_id}) shape 错误: {out.shape}"
        print(f"✓ AdaptiveLoRALinear(crop_id={crop_id}): {out.shape}")

    # 测试不指定 crop_id（平均模式）
    out_avg = adaptive_lora(x)
    assert out_avg.shape == (B, D_out)
    print(f"✓ AdaptiveLoRALinear(平均模式): {out_avg.shape}")

    # 3. 测试参数统计
    stats = adaptive_lora.get_param_stats()
    print(f"✓ 参数统计:")
    for crop_id, info in stats.items():
        if isinstance(info, dict):
            print(f"  crop_{crop_id}: rank={info['rank']}, "
                  f"params={info['params']:,}")
        else:
            print(f"  {crop_id}: {info:,}")

    print("✓ Adaptive LoRA 全部测试通过!\n")


def test_growth_stages():
    """测试 growth_stages.py 中新增的数据结构"""
    from models.growth_stages import (
        CROP_STAGE_ADJACENCY,
        PHENOLOGY_KNOWLEDGE,
        CROP_VISUAL_COMPLEXITY,
    )

    print("=" * 50)
    print("测试 growth_stages 新增数据")
    print("=" * 50)

    # 1. 测试邻接矩阵
    for crop, adj in CROP_STAGE_ADJACENCY.items():
        assert len(adj) == 5, f"{crop} 邻接矩阵行数错误"
        assert all(len(row) == 5 for row in adj), \
            f"{crop} 邻接矩阵列数错误"
        assert adj[0][0] == 1.0, f"{crop} 对角线应该是1.0"
        print(f"✓ {crop} 邻接矩阵: 5x5, "
              f"对角线={adj[0][0]}, 相邻={adj[0][1]}")

    # 2. 测试生育期知识
    for crop, stages in PHENOLOGY_KNOWLEDGE.items():
        assert len(stages) == 5, f"{crop} 阶段数错误"
        for stage, info in stages.items():
            assert 'gdd' in info
            assert 'lai' in info
            assert 'growth_rate' in info
        print(f"✓ {crop} 生育期知识: {len(stages)} 阶段")

    # 3. 测试视觉复杂度
    for crop, info in CROP_VISUAL_COMPLEXITY.items():
        assert 'complexity' in info
        assert 'suggested_rank' in info
        print(f"✓ {crop} 视觉复杂度: {info['complexity']}, "
              f"建议rank={info['suggested_rank']}")

    print("✓ growth_stages 数据 全部测试通过!\n")


def main():
    print("\n" + "=" * 60)
    print("农作物识别创新模块 单元测试")
    print("=" * 60 + "\n")

    try:
        test_growth_stages()
        test_phenology_graph()
        test_confidence_router()
        test_adaptive_lora()

        print("=" * 60)
        print("🎉 全部测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
