"""
ARGUS推理入口 - 完整的检测流程
"""
import torch
import argparse
import json
from typing import Dict, List, Optional
from datetime import datetime

# 导入核心模块
from ARGUS.utils.config import Config
from ARGUS import PhishingFeatureExtractor
from ARGUS import PriorTriggerRules
from ARGUS import LearnableTrigger
from ARGUS import ConfidenceEstimator
from ARGUS import OmniModalFusion
from ARGUS import ConflictDetector
from ARGUS import ConflictResolver
from ARGUS import HierarchicalPromptGenerator
from training.cost_tracker import CostTracker


class ARGUSDetector:
    """
    ARGUS完整检测系统
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Args:
            checkpoint_path: 模型检查点路径（可选）
        """
        self.config = Config()

        # 初始化组件
        self._init_components()

        # 加载模型权重（如果提供）
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        # 成本追踪器
        self.cost_tracker = CostTracker(
            token_price_input=self.config.token_price_input,
            token_price_output=self.config.token_price_output
        )

        print("✅ ARGUS Detector initialized")

    def _init_components(self):
        """初始化所有组件"""
        # 特征提取
        self.feature_extractor = PhishingFeatureExtractor()

        # 任务触发
        self.prior_trigger = PriorTriggerRules(task_names=self.config.TASK_NAMES)
        self.learnable_trigger = LearnableTrigger(
            feature_dim=self.config.num_features,
            num_tasks=self.config.num_tasks,
            prior_weight_init=self.config.prior_weight_init
        ).to(self.config.device)

        # 置信度估计
        self.confidence_estimator = ConfidenceEstimator().to(self.config.device)

        # 冲突检测与消解
        self.conflict_detector = ConflictDetector()
        self.conflict_resolver = ConflictResolver()

        # 多模态融合
        self.multimodal_fusion = OmniModalFusion().to(self.config.device)

        # 提示词生成
        self.prompt_generator = HierarchicalPromptGenerator(
            task_names=self.config.TASK_NAMES
        )

    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        self.learnable_trigger.load_state_dict(checkpoint['trigger_state_dict'])
        self.confidence_estimator.load_state_dict(checkpoint['confidence_state_dict'])
        self.multimodal_fusion.load_state_dict(checkpoint['fusion_state_dict'])

        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")

    def extract_features(self, url: str, html: Optional[str] = None) -> torch.Tensor:
        """提取特征"""
        features = self.feature_extractor.extract_features(url, html)
        return torch.from_numpy(features).float().to(self.config.device)

    def trigger_tasks(self, features: torch.Tensor) -> Dict:
        """触发检查任务"""
        # 先验得分
        prior_scores = self.prior_trigger.compute_trigger_scores(
            features.unsqueeze(0)
        ).squeeze(0)

        # 可学习触发
        with torch.no_grad():
            final_scores, task_indices = self.learnable_trigger.get_top_k_tasks(
                features.unsqueeze(0),
                prior_scores.unsqueeze(0),
                k=self.config.top_k_tasks
            )

        final_scores = final_scores.squeeze(0)
        task_indices = task_indices.squeeze(0)

        # 获取任务名称
        triggered_tasks = [self.config.TASK_NAMES[idx] for idx in task_indices.cpu().numpy()]

        return {
            'triggered_tasks': triggered_tasks,
            'task_scores': final_scores.cpu().numpy(),
            'prior_scores': prior_scores.cpu().numpy()
        }

    def compute_confidences(self, features: torch.Tensor) -> Dict:
        """计算各模态置信度"""
        with torch.no_grad():
            # Quantitative置信度
            quant_conf_dict = self.confidence_estimator.compute_quantitative_confidence(
                features.unsqueeze(0)
            )

            # 模拟Visual和Semantic置信度（实际需要真实计算）
            # 这里简化处理
            confidences = {
                'quantitative': quant_conf_dict['confidence'].squeeze(0),
                'visual': torch.tensor(0.8).to(self.config.device),  # 占位
                'semantic': torch.tensor(0.85).to(self.config.device)  # 占位
            }

        return confidences

    def compute_risk_scores(self, features: torch.Tensor) -> Dict:
        """计算各模态风险评分"""
        # 这里是简化版本，实际需要完整的模型
        # Quantitative: 基于规则特征
        rule_features = features[82:92]  # 规则特征索引82-91
        quant_risk = rule_features.mean()

        # Visual和Semantic（占位）
        risk_scores = {
            'quantitative': quant_risk.unsqueeze(0),
            'visual': torch.tensor(0.5).to(self.config.device),  # 占位
            'semantic': torch.tensor(0.6).to(self.config.device)  # 占位
        }

        return risk_scores

    def detect(self,
               url: str,
               html: Optional[str] = None,
               image: Optional[str] = None,
               use_llm: bool = False) -> Dict:
        """
        完整检测流程

        Args:
            url: 目标URL
            html: HTML内容（可选）
            image: 图像路径（可选）
            use_llm: 是否调用LLM进行最终判断

        Returns:
            result: 检测结果字典
        """
        start_time = datetime.now()

        # 1. 特征提取
        features = self.extract_features(url, html)

        # 2. 任务触发
        task_info = self.trigger_tasks(features)

        # 3. 计算置信度
        confidences = self.compute_confidences(features)

        # 4. 计算风险评分
        risk_scores = self.compute_risk_scores(features)

        # 5. 构建证据字典
        evidence = {
            'quantitative': {
                'features': features.cpu(),
                'domain': url.split('/')[2] if '/' in url else url
            },
            'visual': {
                'brand_topk': [('Unknown', 0.5)],  # 占位
                'logo_confidence': 0.8,
                'manipulation_detected': False
            },
            'semantic': {
                'intent_type': 'unknown',
                'sensitive_action_flags': [],
                'brand_claims': []
            }
        }

        # 6. 冲突检测
        conflicts = self.conflict_detector.detect_conflicts(evidence)

        # 7. 冲突消解
        current_weights = self.multimodal_fusion.get_base_weights()
        conflict_resolution = self.conflict_resolver.resolve_conflicts(
            conflicts, evidence, current_weights
        )

        # 8. 多模态融合
        with torch.no_grad():
            fused_score, fusion_info = self.multimodal_fusion.fuse_with_confidence(
                risk_scores,
                confidences,
                adjusted_weights=conflict_resolution['adjusted_weights']
            )

            # 应用冲突调整
            final_score = torch.clamp(
                fused_score + conflict_resolution['risk_adjustment'],
                0.0, 1.0
            )

        # 9. 生成提示词（如果使用LLM）
        prompt = None
        llm_response = None
        input_tokens = 0
        output_tokens = 0

        if use_llm:
            prompt = self.prompt_generator.generate_prompt(
                triggered_tasks=task_info['triggered_tasks'],
                evidence_dict=evidence,
                conflict_info=conflict_resolution
            )

            # 调用LLM（这里是占位，实际需要API调用）
            # llm_response = call_llm_api(prompt)
            # input_tokens = len(prompt.split())  # 粗略估计
            # output_tokens = len(llm_response.split())

            print(f"\n📝 Generated Prompt Preview (first 500 chars):")
            print(prompt[:500] + "...")

        # 10. 决策
        risk_level = 'HIGH' if final_score > 0.7 else 'MEDIUM' if final_score > 0.4 else 'LOW'

        uncertainty = 1.0 - fusion_info['overall_confidence'] if 'overall_confidence' in fusion_info else 0.5

        if risk_level == 'HIGH':
            action = 'BLOCK'
        elif risk_level == 'MEDIUM':
            action = 'WARN' if uncertainty < 0.3 else 'MANUAL_REVIEW'
        else:
            action = 'ALLOW'

        # 计算耗时
        elapsed_time = (datetime.now() - start_time).total_seconds()

        # 记录成本
        if use_llm:
            self.cost_tracker.log_inference(
                sample_id=url,
                triggered_tasks=task_info['triggered_tasks'],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                inference_time=elapsed_time,
                risk_score=final_score.item()
            )

        # 构建结果
        result = {
            'url': url,
            'risk_score': final_score.item(),
            'risk_level': risk_level,
            'action': action,
            'uncertainty': uncertainty,
            'triggered_tasks': task_info['triggered_tasks'],
            'num_tasks': len(task_info['triggered_tasks']),
            'conflicts_detected': list(conflicts.keys()) if any(conflicts.values()) else [],
            'fusion_info': {
                'available_modalities': fusion_info.get('available_modalities', []),
                'missing_modalities': fusion_info.get('missing_modalities', []),
                'effective_weights': fusion_info.get('effective_weights', {})
            },
            'inference_time_seconds': elapsed_time,
            'prompt': prompt if use_llm else None,
            'llm_response': llm_response
        }

        return result

    def detect_batch(self,
                     urls: List[str],
                     htmls: Optional[List[str]] = None,
                     use_llm: bool = False) -> List[Dict]:
        """批量检测"""
        if htmls is None:
            htmls = [None] * len(urls)

        results = []
        for url, html in zip(urls, htmls):
            result = self.detect(url, html, use_llm=use_llm)
            results.append(result)

        return results

    def print_result(self, result: Dict):
        """美化打印结果"""
        print("\n" + "=" * 80)
        print("ARGUS Detection Result")
        print("=" * 80)

        print(f"\n🌐 URL: {result['url']}")
        print(f"⚠️  Risk Score: {result['risk_score']:.3f}")
        print(f"📊 Risk Level: {result['risk_level']}")
        print(f"🎯 Action: {result['action']}")
        print(f"❓ Uncertainty: {result['uncertainty']:.3f}")

        print(f"\n🔍 Triggered Tasks ({result['num_tasks']}):")
        for task in result['triggered_tasks']:
            print(f"  - {task}")

        if result['conflicts_detected']:
            print(f"\n⚡ Conflicts Detected:")
            for conflict in result['conflicts_detected']:
                print(f"  - {conflict}")

        print(f"\n📡 Modality Status:")
        fusion_info = result['fusion_info']
        print(f"  ✅ Available: {', '.join(fusion_info['available_modalities'])}")
        if fusion_info['missing_modalities']:
            print(f"  ❌ Missing: {', '.join(fusion_info['missing_modalities'])}")

        print(f"\n⏱️  Inference Time: {result['inference_time_seconds']:.3f}s")

        print("=" * 80)


def main(args):
    """主推理流程"""

    print("=" * 80)
    print("ARGUS Phishing Detector - Inference Mode")
    print("=" * 80)

    # 初始化检测器
    detector = ARGUSDetector(checkpoint_path=args.checkpoint)

    # 单个URL检测
    if args.url:
        print(f"\n🔍 Detecting: {args.url}")

        # 读取HTML（如果提供）
        html = None
        if args.html_file:
            with open(args.html_file, 'r', encoding='utf-8') as f:
                html = f.read()

        # 检测
        result = detector.detect(
            url=args.url,
            html=html,
            use_llm=args.use_llm
        )

        # 打印结果
        detector.print_result(result)

        # 保存结果
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 Result saved to {args.output}")

    # 批量检测
    elif args.input_file:
        print(f"\n📁 Batch detection from: {args.input_file}")

        # 读取输入
        with open(args.input_file, 'r') as f:
            data = json.load(f)

        urls = [item['url'] for item in data]
        htmls = [item.get('html', None) for item in data]

        # 检测
        results = detector.detect_batch(urls, htmls, use_llm=args.use_llm)

        # 统计
        high_risk = sum(1 for r in results if r['risk_level'] == 'HIGH')
        medium_risk = sum(1 for r in results if r['risk_level'] == 'MEDIUM')
        low_risk = sum(1 for r in results if r['risk_level'] == 'LOW')

        print(f"\n📊 Batch Results:")
        print(f"  Total: {len(results)}")
        print(f"  HIGH risk: {high_risk}")
        print(f"  MEDIUM risk: {medium_risk}")
        print(f"  LOW risk: {low_risk}")

        # 保存结果
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.output}")

        # 打印成本报告
        if args.use_llm:
            print("\n")
            detector.cost_tracker.print_report()

            # 与baseline比较
            if args.baseline_tokens and args.baseline_cost:
                detector.cost_tracker.compare_with_baseline(
                    baseline_tokens=args.baseline_tokens,
                    baseline_cost=args.baseline_cost
                )

    else:
        print("❌ Error: Please provide --url or --input-file")
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARGUS Phishing Detection - Inference')

    # 输入参数
    parser.add_argument('--url', type=str, default=None,
                        help='Single URL to detect')
    parser.add_argument('--html-file', type=str, default=None,
                        help='HTML file path (for single URL detection)')
    parser.add_argument('--input-file', type=str, default=None,
                        help='JSON file with multiple URLs (batch detection)')

    # 模型参数
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')

    # 推理参数
    parser.add_argument('--use-llm', action='store_true',
                        help='Use LLM for final analysis')
    parser.add_argument('--llm-model', type=str, default='gpt-4-vision-preview',
                        help='LLM model name')

    # 输出参数
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')

    # Baseline比较
    parser.add_argument('--baseline-tokens', type=float, default=None,
                        help='Baseline average tokens (for comparison)')
    parser.add_argument('--baseline-cost', type=float, default=None,
                        help='Baseline average cost (for comparison)')

    args = parser.parse_args()

    main(args)