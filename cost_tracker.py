"""
Cost Tracking Module
Tracks and analyzes token usage and API costs
"""
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import json


class CostTracker:
    """
    Track LLM API costs and token usage

    Compares ARGUS efficiency vs baseline (PhishLLM)
    """

    def __init__(self,
                 token_price_input: float = 0.01,
                 token_price_output: float = 0.03):
        """
        Args:
            token_price_input: Price per 1K input tokens (USD)
            token_price_output: Price per 1K output tokens (USD)
        """
        self.token_price_input = token_price_input
        self.token_price_output = token_price_output

        # Tracking lists
        self.inference_records = []

    def log_inference(self,
                      sample_id: str,
                      triggered_tasks: List[str],
                      input_tokens: int,
                      output_tokens: int,
                      inference_time: float,
                      risk_score: float):
        """
        Log a single inference

        Args:
            sample_id: Sample identifier (URL)
            triggered_tasks: List of triggered task names
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            inference_time: Inference time in seconds
            risk_score: Final risk score
        """
        cost = self.compute_cost(input_tokens, output_tokens)

        record = {
            'timestamp': datetime.now().isoformat(),
            'sample_id': sample_id,
            'triggered_tasks': triggered_tasks,
            'num_tasks': len(triggered_tasks),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost_usd': cost,
            'inference_time': inference_time,
            'risk_score': risk_score
        }

        self.inference_records.append(record)

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Compute cost for token usage

        Args:
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            cost: Cost in USD
        """
        input_cost = (input_tokens / 1000) * self.token_price_input
        output_cost = (output_tokens / 1000) * self.token_price_output
        return input_cost + output_cost

    def get_statistics(self) -> Dict:
        """
        Get overall statistics

        Returns:
            stats: Statistics dictionary
        """
        if not self.inference_records:
            return {'error': 'No records available'}

        # Extract metrics
        num_tasks_list = [r['num_tasks'] for r in self.inference_records]
        input_tokens_list = [r['input_tokens'] for r in self.inference_records]
        output_tokens_list = [r['output_tokens'] for r in self.inference_records]
        total_tokens_list = [r['total_tokens'] for r in self.inference_records]
        costs_list = [r['cost_usd'] for r in self.inference_records]
        times_list = [r['inference_time'] for r in self.inference_records]

        stats = {
            'total_inferences': len(self.inference_records),

            # Task triggering
            'avg_tasks_triggered': float(np.mean(num_tasks_list)),
            'median_tasks_triggered': float(np.median(num_tasks_list)),
            'max_tasks_triggered': int(np.max(num_tasks_list)),
            'min_tasks_triggered': int(np.min(num_tasks_list)),

            # Token usage
            'avg_input_tokens': float(np.mean(input_tokens_list)),
            'avg_output_tokens': float(np.mean(output_tokens_list)),
            'avg_total_tokens': float(np.mean(total_tokens_list)),
            'total_tokens_all': int(np.sum(total_tokens_list)),

            # Costs
            'avg_cost_per_inference': float(np.mean(costs_list)),
            'total_cost': float(np.sum(costs_list)),
            'median_cost': float(np.median(costs_list)),

            # Time
            'avg_inference_time': float(np.mean(times_list)),
            'total_time': float(np.sum(times_list))
        }

        return stats

    def compare_with_baseline(self,
                              baseline_tokens: float,
                              baseline_cost: float) -> Dict:
        """
        Compare with baseline (PhishLLM)

        Args:
            baseline_tokens: Average tokens per inference for baseline
            baseline_cost: Average cost per inference for baseline

        Returns:
            comparison: Comparison metrics
        """
        stats = self.get_statistics()

        if 'error' in stats:
            return stats

        # Token reduction
        token_reduction = (
                                  (baseline_tokens - stats['avg_total_tokens']) / baseline_tokens
                          ) * 100

        # Cost reduction
        cost_reduction = (
                                 (baseline_cost - stats['avg_cost_per_inference']) / baseline_cost
                         ) * 100

        comparison = {
            'baseline_tokens': baseline_tokens,
            'argus_tokens': stats['avg_total_tokens'],
            'token_reduction_percent': token_reduction,

            'baseline_cost': baseline_cost,
            'argus_cost': stats['avg_cost_per_inference'],
            'cost_reduction_percent': cost_reduction,

            'efficiency_gain': token_reduction,  # Primary metric
        }

        return comparison

    def get_task_distribution(self) -> Dict:
        """
        Analyze task triggering distribution

        Returns:
            distribution: Task distribution statistics
        """
        if not self.inference_records:
            return {'error': 'No records available'}

        # Count task frequencies
        task_counts = {}
        for record in self.inference_records:
            for task in record['triggered_tasks']:
                task_counts[task] = task_counts.get(task, 0) + 1

        # Sort by frequency
        sorted_tasks = sorted(
            task_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        total_inferences = len(self.inference_records)

        distribution = {
            'task_frequencies': {
                task: {
                    'count': count,
                    'percentage': (count / total_inferences) * 100
                }
                for task, count in sorted_tasks
            },
            'top_10_tasks': sorted_tasks[:10]
        }

        return distribution

    def print_report(self):
        """Print a formatted cost report"""
        print("\n" + "=" * 80)
        print("ARGUS Cost Analysis Report")
        print("=" * 80)

        stats = self.get_statistics()

        if 'error' in stats:
            print(f"Error: {stats['error']}")
            return

        print(f"\n📊 Overall Statistics:")
        print(f"  Total inferences: {stats['total_inferences']}")
        print(f"  Average tasks triggered: {stats['avg_tasks_triggered']:.2f}")
        print(f"  Task range: {stats['min_tasks_triggered']:.0f} - {stats['max_tasks_triggered']:.0f}")

        print(f"\n🔤 Token Usage:")
        print(f"  Average input tokens: {stats['avg_input_tokens']:.0f}")
        print(f"  Average output tokens: {stats['avg_output_tokens']:.0f}")
        print(f"  Average total tokens: {stats['avg_total_tokens']:.0f}")
        print(f"  Total tokens (all inferences): {stats['total_tokens_all']:.0f}")

        print(f"\n💰 Cost Analysis:")
        print(f"  Average cost per inference: ${stats['avg_cost_per_inference']:.4f}")
        print(f"  Median cost: ${stats['median_cost']:.4f}")
        print(f"  Total cost: ${stats['total_cost']:.4f}")

        print(f"\n⏱️  Performance:")
        print(f"  Average inference time: {stats['avg_inference_time']:.2f}s")
        print(f"  Total time: {stats['total_time']:.2f}s")

        print("\n" + "=" * 80)

    def print_comparison(self, baseline_tokens: float, baseline_cost: float):
        """Print comparison with baseline"""
        comparison = self.compare_with_baseline(baseline_tokens, baseline_cost)

        if 'error' in comparison:
            print(f"Error: {comparison['error']}")
            return

        print("\n" + "=" * 80)
        print("ARGUS vs Baseline Comparison")
        print("=" * 80)

        print(f"\n🔤 Token Usage:")
        print(f"  Baseline: {comparison['baseline_tokens']:.0f} tokens")
        print(f"  ARGUS: {comparison['argus_tokens']:.0f} tokens")
        print(f"  Reduction: {comparison['token_reduction_percent']:.1f}%")

        print(f"\n💰 Cost:")
        print(f"  Baseline: ${comparison['baseline_cost']:.4f}")
        print(f"  ARGUS: ${comparison['argus_cost']:.4f}")
        print(f"  Reduction: {comparison['cost_reduction_percent']:.1f}%")

        print(f"\n📈 Efficiency Gain: {comparison['efficiency_gain']:.1f}%")

        print("\n" + "=" * 80)

    def save_records(self, filepath: str):
        """
        Save records to JSON file

        Args:
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(self.inference_records, f, indent=2)

        print(f"💾 Records saved to {filepath}")

    def load_records(self, filepath: str):
        """
        Load records from JSON file

        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            self.inference_records = json.load(f)

        print(f"✅ Loaded {len(self.inference_records)} records from {filepath}")