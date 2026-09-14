"""Exact parameter-level gradient diagnostics on synthetic inputs only."""
import json

import pytest
import torch

from argus import ConfidenceEstimator, LearnableTrigger, OmniModalFusion
from argus.training.trainer import ARGUSTrainer
from argus.utils.config import Config


def diagnostic(loss_name):
    torch.manual_seed(7)
    config = Config()
    config.device = torch.device('cpu')
    trainer = ARGUSTrainer(LearnableTrigger(num_tasks=config.num_tasks),
                           ConfidenceEstimator(), OmniModalFusion(), config)
    features = torch.zeros(2, 100)
    features[:, 82:87] = 0.2
    terms = trainer.compute_loss_terms(features, torch.full((2, 17), 0.4), torch.tensor([0, 1]))
    terms[loss_name].backward()
    result = {'finite_nonzero': [], 'zero': [], 'disconnected': []}
    for prefix, module in [('trigger', trainer.trigger), ('confidence', trainer.confidence), ('fusion', trainer.fusion)]:
        for name, parameter in module.named_parameters():
            name = f'{prefix}.{name}'
            gradient = parameter.grad
            if gradient is None:
                result['disconnected'].append(name)
            else:
                assert torch.isfinite(gradient).all(), name
                result['finite_nonzero' if gradient.abs().sum() > 0 else 'zero'].append(name)
    return result


@pytest.mark.parametrize('loss_name', [
    'detection_loss', 'efficiency_penalty', 'relevance_reward', 'auxiliary_loss', 'total_loss',
])
def test_gradient_diagnostics(loss_name):
    result = diagnostic(loss_name)
    print('\n' + loss_name + ': ' + json.dumps(result, sort_keys=True))
    expected_prefixes = ({'confidence', 'fusion'} if loss_name == 'detection_loss' else
                         {'trigger', 'confidence', 'fusion'} if loss_name == 'total_loss' else {'trigger'})
    assert {name.split('.')[0] for name in result['finite_nonzero']} == expected_prefixes
    assert not result['zero']
    # Every parameter tensor of each expected component receives a nonzero gradient.
    assert not any(name.split('.')[0] in expected_prefixes for name in result['disconnected'])


@pytest.mark.xfail(strict=True, reason='Current task selection does not condition detection; research redesign deferred')
def test_trigger_is_detection_loss_connected():
    result = diagnostic('detection_loss')
    assert any(name.startswith('trigger.') for name in result['finite_nonzero'])
