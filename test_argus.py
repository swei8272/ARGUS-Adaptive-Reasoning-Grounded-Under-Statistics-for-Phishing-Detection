"""
快速测试ARGUS系统的各个组件
"""
import torch
from ARGUS.core.feature_extractor import PhishingFeatureExtractor
from ARGUS.core.prior_trigger import PriorTriggerRules
from ARGUS.core.learnable_trigger import LearnableTrigger
from ARGUS.core.multimodal_fusion import OmniModalFusion
from ARGUS.core.conflict_detector import ConflictDetector
from ARGUS.core.conflict_resolver import ConflictResolver
from ARGUS.core.prompt_generator import HierarchicalPromptGenerator
from ARGUS.utils.config import Config


def test_feature_extraction():
    """测试特征提取"""
    print("\n" + "=" * 80)
    print("Test 1: Feature Extraction")
    print("=" * 80)

    extractor = PhishingFeatureExtractor()

    # 测试URL
    test_urls = [
        "https://paypa1-verify.tk/urgent-login?id=123",
        "https://www.paypal.com/signin",
        "http://192.168.1.1/admin"
    ]

    for url in test_urls:
        features = extractor.extract_features(url)
        print(f"\nURL: {url}")
        print(f"  Feature shape: {features.shape}")
        print(f"  Domain entropy: {features[17]:.2f}")
        print(f"  Suspicious TLD: {features[19]}")
        print(f"  Typosquatting rule: {features[82]}")


def test_task_triggering():
    """测试任务触发"""
    print("\n" + "=" * 80)
    print("Test 2: Task Triggering")
    print("=" * 80)

    config = Config()
    extractor = PhishingFeatureExtractor()
    prior_trigger = PriorTriggerRules(task_names=config.TASK_NAMES)
    learnable_trigger = LearnableTrigger(
        feature_dim=100,
        num_tasks=len(config.TASK_NAMES)
    )

    url = "https://paypa1.tk/login"
    features = torch.from_numpy(extractor.extract_features(url)).float()

    # 先验触发
    prior_scores = prior_trigger.compute_trigger_scores(features.unsqueeze(0))
    print(f"\nPrior trigger scores shape: {prior_scores.shape}")
    print(f"Top triggered tasks (prior):")
    top_k = torch.topk(prior_scores[0], k=5)
    for score, idx in zip(top_k.values, top_k.indices):
        print(f"  {config.TASK_NAMES[idx]}: {score:.3f}")

    # 可学习触发
    final_scores, task_indices = learnable_trigger.get_top_k_tasks(
        features.unsqueeze(0),
        prior_scores,
        k=6
    )
    print(f"\nTop-6 triggered tasks (learned):")
    for score, idx in zip(final_scores[0], task_indices[0]):
        print(f"  {config.TASK_NAMES[idx]}: {score:.3f}")


def test_omnimodal_fusion():
    """测试Omni-Modal融合"""
    print("\n" + "=" * 80)
    print("Test 3: Omni-Modal Fusion")
    print("=" * 80)

    fusion = OmniModalFusion()

    # 场景1：所有模态可用
    print("\nScenario 1: All modalities available")
    risk_scores = {
        'quantitative': torch.tensor([0.8]),
        'visual': torch.tensor([0.6]),
        'semantic': torch.tensor([0.7])
    }
    confidences = {
        'quantitative': torch.tensor([0.9]),
        'visual': torch.tensor([0.8]),
        'semantic': torch.tensor([0.85])
    }

    fused, info = fusion.fuse_with_confidence(risk_scores, confidences)
    print(f"Fused score: {fused.item():.3f}")
    print(fusion.explain_fusion(info))

    # 场景2：视觉缺失
    print("\n" + "-" * 80)
    print("Scenario 2: Visual modality missing")
    risk_scores_partial = {
        'quantitative': torch.tensor([0.8]),
        'semantic': torch.tensor([0.7])
    }
    confidences_partial = {
        'quantitative': torch.tensor([0.9]),
        'semantic': torch.tensor([0.85])
    }

    fused, info = fusion.fuse_with_confidence(risk_scores_partial, confidences_partial)
    print(f"Fused score: {fused.item():.3f}")
    print(fusion.explain_fusion(info))


def test_conflict_detection():
    """测试冲突检测"""
    print("\n" + "=" * 80)
    print("Test 4: Conflict Detection & Resolution")
    print("=" * 80)

    detector = ConflictDetector()
    resolver = ConflictResolver()

    # 构建测试证据
    evidence = {
        'quantitative': {
            'features': torch.zeros(100),
            'domain': 'paypa1.tk'
        },
        'visual': {
            'brand_topk': [('PayPal', 0.9)],
            'logo_confidence': 0.85,
            'manipulation_detected': False
        },
        'semantic': {
            'intent_type': 'login',
            'brand_claims': ['PayPal']
        }
    }

    # 手动设置一些特征值
    evidence['quantitative']['features'][40] = 1.0  # 密码字段
    evidence['quantitative']['features'][43] = 1.0  # 外部action

    # 检测冲突
    conflicts = detector.detect_conflicts(evidence)
    print(f"\nDetected conflicts:")
    for conflict_type, detected in conflicts.items():
        if detected:
            print(f"  ✅ {conflict_type}")

    # 消解冲突
    current_weights = {'quantitative': 0.4, 'visual': 0.3, 'semantic': 0.3}
    resolution = resolver.resolve_conflicts(conflicts, evidence, current_weights)

    print(f"\nResolution:")
    print(f"  Adjusted weights: {resolution['adjusted_weights']}")
    print(f"  Risk adjustment: {resolution['risk_adjustment']}")
    print(f"  Additional tasks: {resolution['additional_tasks']}")


def test_prompt_generation():
    """测试提示词生成"""
    print("\n" + "=" * 80)
    print("Test 5: Prompt Generation")
    print("=" * 80)

    config = Config()
    generator = HierarchicalPromptGenerator(task_names=config.TASK_NAMES)

    # 测试数据
    triggered_tasks = [
        'typosquatting_detection',
        'credential_harvesting_check',
        'domain_age_verification'
    ]

    evidence_dict = {
        'quantitative': {
            'features': torch.zeros(100)
        },
        'visual': {
            'brand_topk': [('PayPal', 0.9)],
            'logo_confidence': 0.85
        },
        'semantic': {
            'intent_type': 'login',
            'brand_claims': ['PayPal']
        }
    }

    conflict_info = {
        'conflicts_detected': ['brand_domain_mismatch'],
        'resolution_log': ['Downweight visual evidence']
    }

    # 生成提示词
    prompt = generator.generate_prompt(triggered_tasks, evidence_dict, conflict_info)

    print(f"\nGenerated prompt preview (first 800 chars):")
    print(prompt[:800])
    print("...")
    print(f"\nTotal prompt length: {len(prompt)} characters")


def main():
    """运行所有测试"""
    print("=" * 80)
    print("ARGUS System Component Tests")
    print("=" * 80)

    test_feature_extraction()
    test_task_triggering()
    test_omnimodal_fusion()
    test_conflict_detection()
    test_prompt_generation()

    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()