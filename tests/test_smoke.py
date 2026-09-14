"""CPU component checks with synthetic fixtures, not ARGUS performance results."""
import importlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from argus import (
    ConflictDetector, ConflictResolver, HierarchicalPromptGenerator,
    LearnableTrigger, OmniModalFusion, PhishingFeatureExtractor, PriorTriggerRules,
)
from argus.utils.config import Config


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    # Exercise the existing WHOIS-unavailable path, never live network evidence.
    monkeypatch.setattr('argus.core.feature_extractor.WHOIS_AVAILABLE', False)
    monkeypatch.setattr(Config, 'device', torch.device('cpu'))
    torch.manual_seed(0)


@pytest.mark.parametrize('url', [
    'https://paypa1-verify.tk/urgent-login?id=123',
    'https://www.paypal.com/signin',
    'http://192.168.1.1/admin',
])
def test_feature_extraction(url):
    features = PhishingFeatureExtractor().extract_features(url)
    assert features.shape == (100,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_prior_and_learnable_trigger():
    features = torch.from_numpy(np.stack([
        PhishingFeatureExtractor().extract_features(url)
        for url in ['https://paypa1.tk/login', 'https://example.com/']
    ]))
    prior = PriorTriggerRules(Config.TASK_NAMES).compute_trigger_scores(features)
    assert prior.shape == (2, len(Config.TASK_NAMES))
    assert torch.isfinite(prior).all()
    assert ((prior >= 0) & (prior <= 1)).all()
    trigger = LearnableTrigger(feature_dim=100, num_tasks=len(Config.TASK_NAMES)).cpu().eval()
    with torch.no_grad():
        scores, indices = trigger.get_top_k_tasks(features, prior, k=6)
        all_scores = trigger(features, prior)
    assert scores.shape == indices.shape == (2, 6)
    assert indices.dtype == torch.int64
    assert ((indices >= 0) & (indices < len(Config.TASK_NAMES))).all()
    assert all(row.unique().numel() == 6 for row in indices)
    assert torch.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert (scores[:, :-1] >= scores[:, 1:]).all()
    torch.testing.assert_close(scores, all_scores.gather(1, indices))


@pytest.mark.parametrize('missing', [None, 'visual'])
def test_fusion_cpu(missing):
    fusion = OmniModalFusion().cpu().eval()
    risks = dict(zip(['quantitative', 'visual', 'semantic'],
                     [torch.tensor([0.8]), torch.tensor([0.6]), torch.tensor([0.7])]))
    confidences = {name: torch.tensor([1.0]) for name in risks}
    if missing:
        del risks[missing]
        del confidences[missing]
    with torch.no_grad():
        fused, info = fusion.fuse_with_confidence(risks, confidences)
    assert fused.shape == (1,)
    assert fused.device.type == 'cpu'
    assert torch.isfinite(fused).all()
    torch.testing.assert_close(fused, torch.stack(list(risks.values())).mean(0))
    assert info['missing_modalities'] == ([missing] if missing else [])


def test_conflict_detection_and_resolution():
    features = torch.zeros(100)
    features[40] = features[43] = 1.0
    evidence = {
        'quantitative': {'features': features, 'domain': 'paypa1.tk'},
        'visual': {'brand_topk': [('PayPal', 0.9)], 'logo_confidence': 0.85},
        'semantic': {'intent_type': 'login', 'brand_claims': ['PayPal']},
    }
    conflicts = ConflictDetector().detect_conflicts(evidence)
    assert conflicts['brand_domain_mismatch']
    assert conflicts['credential_harvesting_external_action']
    weights = {'quantitative': 0.4, 'visual': 0.3, 'semantic': 0.3}
    resolver = ConflictResolver()
    resolution = resolver.resolve_conflicts(conflicts, evidence, weights)
    assert resolution['risk_adjustment'] == 1.0
    assert resolution['resolution_log']
    assert sum(resolution['adjusted_weights'].values()) == pytest.approx(1.0)
    assert resolution['adjusted_weights'] == weights
    no_conflict = resolver.resolve_conflicts({}, {}, weights)
    assert no_conflict['risk_adjustment'] == 0.0
    assert no_conflict['adjusted_weights'] == weights
    assert no_conflict['additional_tasks'] == []


def test_prompt_generation():
    prompt = HierarchicalPromptGenerator(Config.TASK_NAMES).generate_prompt(
        ['typosquatting_detection', 'credential_harvesting_check'],
        {'quantitative': {'features': torch.zeros(100)},
         'visual': {'brand_topk': [('PayPal', 0.9)], 'logo_confidence': 0.85},
         'semantic': {'intent_type': 'login', 'brand_claims': ['PayPal']}},
        {'conflicts_detected': ['brand_domain_mismatch'],
         'resolution_log': ['Synthetic conflict fixture']},
    )
    assert isinstance(prompt, str)
    assert prompt.strip()
    assert 'PayPal' in prompt


@pytest.mark.parametrize('module', ['argus.train', 'argus.inference'])
def test_entry_point_imports(module):
    assert callable(importlib.import_module(module).main)


def test_inference_help_outside_checkout(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'argus.inference', '--help'],
                            cwd=tmp_path, text=True, capture_output=True, check=True)
    assert 'ARGUS Phishing Detection' in result.stdout
    assert '--url' in result.stdout
