"""
先验触发器 - 基于规则的任务触发
"""
import torch
from typing import Dict, List, Optional
from ARGUS.core.constants import PRIOR_TRIGGER_RULES, FEATURE_INDEX_MAP


class PriorTriggerRules:
    """基于领域知识的先验触发规则"""

    def __init__(self, task_names: List[str]):
        """
        Args:
            task_names: 任务名称列表
        """
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.rules = PRIOR_TRIGGER_RULES

        # 验证：检查是否所有任务都有规则定义
        self._validate_rules()

    def _validate_rules(self):
        """验证规则完整性"""
        missing_rules = []
        for task_name in self.task_names:
            if task_name not in self.rules:
                missing_rules.append(task_name)

        if missing_rules:
            import warnings
            warnings.warn(
                f"Following tasks have no prior rules defined: {missing_rules}. "
                f"They will have default trigger score of 0.0",
                stacklevel=2
            )

    def _evaluate_condition(self,
                           features: torch.Tensor,
                           feature_name: str,
                           operator: str,
                           threshold: float) -> torch.Tensor:
        """
        评估单个条件

        Args:
            features: (batch_size, num_features) 特征张量
            feature_name: 特征名称
            operator: 比较操作符
            threshold: 阈值

        Returns:
            result: (batch_size,) 条件结果 [0.0/1.0]
        """
        # 检查特征是否存在于映射中
        if feature_name not in FEATURE_INDEX_MAP:
            # 特征不存在，返回全0
            return torch.zeros(features.shape[0], device=features.device)

        feat_idx = FEATURE_INDEX_MAP[feature_name]

        # 检查索引是否越界
        if feat_idx >= features.shape[1]:
            return torch.zeros(features.shape[0], device=features.device)

        feat_values = features[:, feat_idx]

        # 应用比较操作
        if operator == '>':
            result = (feat_values > threshold).float()
        elif operator == '>=':
            result = (feat_values >= threshold).float()
        elif operator == '==':
            result = (feat_values == threshold).float()
        elif operator == '<':
            result = (feat_values < threshold).float()
        elif operator == '<=':
            result = (feat_values <= threshold).float()
        elif operator == '!=':
            result = (feat_values != threshold).float()
        else:
            # 未知操作符，返回全0
            result = torch.zeros_like(feat_values)

        return result

    def _aggregate_conditions(self,
                             condition_results: torch.Tensor,
                             aggregation: str) -> torch.Tensor:
        """
        聚合多个条件的结果

        Args:
            condition_results: (batch_size, num_conditions) 条件结果
            aggregation: 聚合方式 ('any', 'all', 'mean', 'weighted')

        Returns:
            aggregated: (batch_size,) 聚合后的得分
        """
        if condition_results.shape[1] == 0:
            # 没有条件，返回全0
            return torch.zeros(condition_results.shape[0], device=condition_results.device)

        if aggregation == 'any':
            # 任一条件满足（逻辑OR）
            return condition_results.max(dim=1)[0]

        elif aggregation == 'all':
            # 所有条件满足（逻辑AND）
            return condition_results.min(dim=1)[0]

        elif aggregation == 'mean':
            # 条件平均
            return condition_results.mean(dim=1)

        elif aggregation == 'weighted':
            # 加权平均（权重递减）
            weights = torch.linspace(1.0, 0.5, condition_results.shape[1], device=condition_results.device)
            weights = weights / weights.sum()
            return (condition_results * weights.unsqueeze(0)).sum(dim=1)

        else:
            # 默认：平均
            return condition_results.mean(dim=1)

    def compute_trigger_scores(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算先验触发得分

        Args:
            features: (batch_size, num_features) 特征向量

        Returns:
            trigger_scores: (batch_size, num_tasks) 触发得分 [0, 1]
        """
        batch_size = features.shape[0]
        device = features.device

        scores = torch.zeros(batch_size, self.num_tasks, device=device)

        # 遍历每个任务
        for task_idx, task_name in enumerate(self.task_names):
            if task_name not in self.rules:
                # 如果没有定义规则，默认得分为0
                continue

            rule = self.rules[task_name]
            conditions = rule.get('conditions', [])
            aggregation = rule.get('aggregation', 'any')

            if not conditions:
                # 没有条件定义
                continue

            # 评估每个条件
            condition_results = []
            for condition in conditions:
                if len(condition) != 3:
                    # 条件格式错误，跳过
                    continue

                feature_name, operator, threshold = condition
                result = self._evaluate_condition(features, feature_name, operator, threshold)
                condition_results.append(result)

            if not condition_results:
                # 没有有效条件
                continue

            # 聚合条件结果
            condition_results = torch.stack(condition_results, dim=1)  # (batch, num_conditions)
            scores[:, task_idx] = self._aggregate_conditions(condition_results, aggregation)

        return scores

    def get_triggered_tasks(self,
                           features: torch.Tensor,
                           threshold: float = 0.5) -> List[List[str]]:
        """
        获取触发的任务列表

        Args:
            features: (batch_size, num_features)
            threshold: 触发阈值

        Returns:
            triggered_tasks: List of List[str], 每个样本触发的任务名称
        """
        scores = self.compute_trigger_scores(features)
        triggered = scores > threshold

        result = []
        for i in range(features.shape[0]):
            tasks = [self.task_names[j] for j in range(self.num_tasks) if triggered[i, j]]
            result.append(tasks)

        return result

    def get_task_score(self,
                      features: torch.Tensor,
                      task_name: str) -> Optional[torch.Tensor]:
        """
        获取特定任务的触发得分

        Args:
            features: (batch_size, num_features)
            task_name: 任务名称

        Returns:
            score: (batch_size,) 或 None（如果任务不存在）
        """
        if task_name not in self.task_names:
            return None

        task_idx = self.task_names.index(task_name)
        scores = self.compute_trigger_scores(features)
        return scores[:, task_idx]

    def get_rule_explanation(self, task_name: str) -> str:
        """
        获取任务规则的文字解释

        Args:
            task_name: 任务名称

        Returns:
            explanation: 规则解释
        """
        if task_name not in self.rules:
            return f"No rule defined for task: {task_name}"

        rule = self.rules[task_name]
        conditions = rule.get('conditions', [])
        aggregation = rule.get('aggregation', 'any')

        if not conditions:
            return f"Task '{task_name}' has empty conditions"

        lines = [f"Task: {task_name}"]
        lines.append(f"Aggregation: {aggregation.upper()}")
        lines.append("Conditions:")

        for i, condition in enumerate(conditions, 1):
            if len(condition) == 3:
                feature_name, operator, threshold = condition
                lines.append(f"  {i}. {feature_name} {operator} {threshold}")
            else:
                lines.append(f"  {i}. Invalid condition format")

        return "\n".join(lines)

    def analyze_features(self, features: torch.Tensor, sample_idx: int = 0) -> Dict:
        """
        分析单个样本的特征和触发情况

        Args:
            features: (batch_size, num_features)
            sample_idx: 样本索引

        Returns:
            analysis: 分析结果字典
        """
        if sample_idx >= features.shape[0]:
            return {'error': f'Sample index {sample_idx} out of range'}

        # 计算触发得分
        scores = self.compute_trigger_scores(features)
        sample_scores = scores[sample_idx]

        # 获取触发的任务
        triggered_tasks = []
        not_triggered_tasks = []

        for task_idx, task_name in enumerate(self.task_names):
            score = sample_scores[task_idx].item()
            if score > 0.5:
                triggered_tasks.append((task_name, score))
            else:
                not_triggered_tasks.append((task_name, score))

        # 按得分排序
        triggered_tasks.sort(key=lambda x: x[1], reverse=True)
        not_triggered_tasks.sort(key=lambda x: x[1], reverse=True)

        # 提取关键特征值
        key_features = {}
        for feature_name, feat_idx in FEATURE_INDEX_MAP.items():
            if feat_idx < features.shape[1]:
                key_features[feature_name] = features[sample_idx, feat_idx].item()

        return {
            'sample_index': sample_idx,
            'triggered_tasks': triggered_tasks,
            'not_triggered_tasks': not_triggered_tasks,
            'key_features': key_features,
            'total_tasks': self.num_tasks,
            'num_triggered': len(triggered_tasks)
        }

    def print_analysis(self, features: torch.Tensor, sample_idx: int = 0):
        """
        打印样本分析（用于调试）

        Args:
            features: (batch_size, num_features)
            sample_idx: 样本索引
        """
        analysis = self.analyze_features(features, sample_idx)

        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            return

        print("=" * 80)
        print(f"Prior Trigger Analysis - Sample {sample_idx}")
        print("=" * 80)

        print(f"\n✅ Triggered Tasks ({analysis['num_triggered']}/{analysis['total_tasks']}):")
        for task_name, score in analysis['triggered_tasks']:
            print(f"  {task_name}: {score:.3f}")

        print(f"\n❌ Not Triggered Tasks:")
        for task_name, score in analysis['not_triggered_tasks'][:5]:  # 只显示前5个
            print(f"  {task_name}: {score:.3f}")

        print(f"\n📊 Key Features (sample):")
        important_features = [
            'domain_entropy', 'is_suspicious_tld', 'num_subdomains',
            'has_ip_address', 'punycode_domain', 'num_password_fields',
            'form_action_external', 'rule_typosquatting'
        ]
        for feature_name in important_features:
            if feature_name in analysis['key_features']:
                value = analysis['key_features'][feature_name]
                print(f"  {feature_name}: {value:.2f}")

        print("=" * 80)

    def get_statistics(self, features: torch.Tensor) -> Dict:
        """
        获取批量触发统计

        Args:
            features: (batch_size, num_features)

        Returns:
            stats: 统计信息
        """
        scores = self.compute_trigger_scores(features)
        triggered = (scores > 0.5).float()

        # 每个样本触发的任务数
        tasks_per_sample = triggered.sum(dim=1)

        # 每个任务被触发的次数
        triggers_per_task = triggered.sum(dim=0)

        # 最常被触发的任务
        top_tasks = []
        for task_idx in torch.argsort(triggers_per_task, descending=True):
            task_name = self.task_names[task_idx]
            count = triggers_per_task[task_idx].item()
            if count > 0:
                top_tasks.append((task_name, int(count)))

        return {
            'batch_size': features.shape[0],
            'avg_tasks_per_sample': tasks_per_sample.mean().item(),
            'max_tasks_triggered': tasks_per_sample.max().item(),
            'min_tasks_triggered': tasks_per_sample.min().item(),
            'top_triggered_tasks': top_tasks,
            'total_triggers': triggered.sum().item()
        }

    def print_statistics(self, features: torch.Tensor):
        """打印批量统计（用于调试）"""
        stats = self.get_statistics(features)

        print("=" * 80)
        print("Prior Trigger Statistics")
        print("=" * 80)

        print(f"\n📊 Batch Statistics:")
        print(f"  Batch size: {stats['batch_size']}")
        print(f"  Avg tasks per sample: {stats['avg_tasks_per_sample']:.2f}")
        print(f"  Max tasks triggered: {stats['max_tasks_triggered']:.0f}")
        print(f"  Min tasks triggered: {stats['min_tasks_triggered']:.0f}")
        print(f"  Total triggers: {stats['total_triggers']:.0f}")

        print(f"\n🔥 Top Triggered Tasks:")
        for task_name, count in stats['top_triggered_tasks'][:10]:
            percentage = (count / stats['batch_size']) * 100
            print(f"  {task_name}: {count} times ({percentage:.1f}%)")

        print("=" * 80)