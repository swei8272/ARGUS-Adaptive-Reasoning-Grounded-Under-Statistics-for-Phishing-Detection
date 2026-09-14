"""Mac deployment checks; availability mocks never count as real MPS execution."""
import os
from unittest.mock import Mock

import pytest
import torch

from argus.utils import config


@pytest.mark.parametrize('mps,cuda,expected', [(True, True, 'mps'), (True, False, 'mps'),
                                             (False, True, 'cuda'), (False, False, 'cpu')])
def test_device_priority(mps, cuda, expected, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: mps)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda)
    assert config._select_device().type == expected


@pytest.mark.parametrize('backend', ['cpu', 'mps', 'cuda'])
def test_seeding_calls_only_selected_accelerator(backend, monkeypatch):
    from argus.utils.runtime import seed_everything
    mps_seed, cuda_seed = Mock(), Mock()
    monkeypatch.setattr(torch.mps, 'manual_seed', mps_seed)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', cuda_seed)
    seed_everything(17, torch.device(backend))
    assert mps_seed.call_count == (backend == 'mps')
    assert cuda_seed.call_count == (backend == 'cuda')


def test_cpu_seed_repeatability():
    import random
    import numpy as np
    from argus.utils.runtime import seed_everything
    seed_everything(17, torch.device('cpu'))
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_everything(17, torch.device('cpu'))
    second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2])


def test_mps_dtype_policy_without_hardware():
    from argus.utils.runtime import device_dtype
    assert device_dtype(torch.device('mps'), torch.float64) == torch.float32
    assert device_dtype(torch.device('mps'), torch.float16) == torch.float32
    assert device_dtype(torch.device('cpu'), torch.float64) == torch.float64


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Real MPS hardware unavailable')
def test_real_mps_tensor_and_argus_backward(monkeypatch):
    from argus import ConfidenceEstimator, LearnableTrigger, OmniModalFusion
    from argus.training.trainer import ARGUSTrainer
    from argus.utils.runtime import seed_everything
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '1', 'Disable silent CPU fallback for this test'
    seed_everything(7, torch.device('mps'))
    cfg = config.Config()
    cfg.device = torch.device('mps')
    trainer = ARGUSTrainer(LearnableTrigger(num_tasks=cfg.num_tasks),
                           ConfidenceEstimator(), OmniModalFusion(), cfg)
    # Exercise CPU float64 -> selected MPS float32 conversion before dispatch.
    features = torch.zeros(2, 100, dtype=torch.float64)
    features[:, 82:87] = 0.2
    terms = trainer.compute_loss_terms(features, torch.full((2, 17), 0.4, dtype=torch.float64),
                                       torch.tensor([0, 1]))
    assert terms['probabilities'].device.type == 'mps'
    assert terms['probabilities'].dtype == torch.float32
    terms['total_loss'].backward()
    for module in [trainer.trigger, trainer.confidence, trainer.fusion]:
        for parameter in module.parameters():
            assert parameter.device.type == 'mps' and parameter.dtype == torch.float32
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    torch.mps.synchronize()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Real MPS hardware unavailable')
@pytest.mark.parametrize('case', ['all', 'missing', 'scalar', 'reshape', 'fallback'])
def test_real_mps_fusion_paths(case):
    from argus import OmniModalFusion
    model = OmniModalFusion().to(device='mps', dtype=torch.float32).eval()
    shape = () if case == 'scalar' else (2, 1) if case == 'reshape' else (2,)
    risks = {name: torch.full(shape, value, device='mps', dtype=torch.float32, requires_grad=True)
             for name, value in zip(model.modality_names, [0.1, 0.4, 0.9])}
    confs = {name: torch.full(shape, 0.0 if case == 'fallback' else 0.8, device='mps') for name in risks}
    if case == 'missing':
        del risks['visual']
        del confs['visual']
    if case == 'fallback':
        with pytest.warns(UserWarning, match='No available modalities'):
            result, _ = model.fuse_with_confidence(risks, confs)
    else:
        result, _ = model.fuse_with_confidence(risks, confs)
        result.sum().backward()
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in risks.values())
    assert result.device.type == 'mps' and result.dtype == torch.float32
    assert torch.isfinite(result).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Real MPS hardware unavailable')
def test_real_mps_training_step_and_inference(monkeypatch):
    from argus import ConfidenceEstimator, LearnableTrigger, OmniModalFusion
    from argus.inference import ARGUSDetector
    from argus.training.trainer import ARGUSTrainer
    monkeypatch.setattr(config.Config, 'device', torch.device('mps'))
    cfg = config.Config()
    trainer = ARGUSTrainer(LearnableTrigger(num_tasks=17), ConfidenceEstimator(), OmniModalFusion(), cfg)
    features = torch.zeros(2, 100, dtype=torch.float64)
    features[:, 82:87] = 0.2
    before = trainer.fusion.weight_logits['quantitative'].detach().clone()
    # Unit fixture only: do not print or claim phishing performance metrics.
    report = trainer.train_epoch([{'features': features,
        'prior_scores': torch.full((2, 17), 0.4, dtype=torch.float64), 'label': torch.tensor([0, 1])}], 1)
    assert report['data_report']['attempted_samples'] == 2
    assert not torch.equal(before, trainer.fusion.weight_logits['quantitative'])
    monkeypatch.setattr('argus.core.feature_extractor.WHOIS_AVAILABLE', False)
    detector = ARGUSDetector()
    result = detector.detect('https://example.com/', image='/unused-placeholder.png')
    assert result['llm_status'] == 'not_requested'
    assert result['visual_status'] == 'placeholder_image_unused'
    assert isinstance(result['uncertainty'], float)
    for model in [detector.learnable_trigger, detector.confidence_estimator, detector.multimodal_fusion]:
        assert all(p.device.type == 'mps' and p.dtype == torch.float32 for p in model.parameters())
    import json
    json.dumps(result)  # No device tensors leak into the API response.
