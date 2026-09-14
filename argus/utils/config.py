"""
配置文件
"""
import torch


class Config:
    # ============ 设备配置 ============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============ 特征配置 ============
    num_features = 100
    num_tasks = 17  # Currently 17 detection tasks (0-16), with 3 visual tasks planned

    # ============ 任务定义 ============
    TASK_NAMES = [
        # URL基础统计桶任务 (0-3)
        'domain_brand_consistency',
        'high_entropy_subdomain_check',
        'ip_punycode_risk_check',
        'domain_age_verification',

        # HTML表单与资源桶任务 (4-7)
        'form_action_consistency',
        'credential_harvesting_check',
        'external_resource_check',
        'redirect_chain_check',

        # 特征交互桶任务 (8-11)
        'interaction_evidence_reinforcement',
        'login_external_link_priority',

        # 领域规则桶任务 (12-17)
        'typosquatting_detection',
        'homograph_attack_detection',
        'phishing_kit_pattern',
        'credential_form_external_action',
        'complex_subdomain_structure',
        'short_high_entropy_domain',

        # 时序桶任务 (18)
        'temporal_consistency_check',

        # 视觉任务 (19-21) - 稍后扩展
        # 'visual_manipulation_detection',
        # 'brand_domain_mismatch',
        # 'layout_consistency_check',
    ]

    # ============ 触发器配置 ============
    # 先验权重（初始值）
    prior_weight_init = 0.7

    # Top-K任务选择
    top_k_tasks = 6

    # ============ 置信度配置 ============
    # 置信度阈值
    confidence_thresholds = {
        'high': 0.8,
        'medium': 0.5,
        'low': 0.3
    }

    # ============ 冲突优先级 ============
    CONFLICT_PRIORITY = {
        # P0: 绝对优先
        'credential_harvesting_external_action': {
            'priority': 0,
            'action': 'force_high_risk',
            'override': True
        },
        # P1: 品牌相关
        'brand_domain_mismatch': {
            'priority': 1,
            'action': 'downweight_visual',
            'add_tasks': ['domain_brand_consistency']
        },
        # P2: 视觉质量
        'visual_semantic_mismatch': {
            'priority': 2,
            'action': 'increase_uncertainty',
        },
        # P3: 时序异常
        'new_domain_old_brand': {
            'priority': 3,
            'action': 'raise_suspicion',
        }
    }

    # ============ 训练配置 ============
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-5

    # 损失权重
    detection_loss_weight = 1.0
    efficiency_penalty_weight = 0.1
    relevance_reward_weight = 0.5

    # ============ 推理配置 ============
    # LLM API配置（需要自己填写）
    llm_api_key = "YOUR_API_KEY"
    llm_model = "gpt-4-vision-preview"  # 或 "claude-3-opus-20240229"
    llm_max_tokens = 1000
    llm_temperature = 0.1

    # ============ 成本配置 ============
    # Token价格（美元/1K tokens）
    token_price_input = 0.01
    token_price_output = 0.03