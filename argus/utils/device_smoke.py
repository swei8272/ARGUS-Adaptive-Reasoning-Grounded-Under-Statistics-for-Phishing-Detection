"""Run real selected-device operations on synthetic inputs, without model metrics."""
import json
import os

import torch

from argus import ConfidenceEstimator, LearnableTrigger, OmniModalFusion
from argus.training.trainer import ARGUSTrainer
from argus.utils.config import Config
from argus.utils.runtime import seed_everything


def main():
    config = Config()
    report = {
        'mps_is_built': torch.backends.mps.is_built(),
        'mps_is_available': torch.backends.mps.is_available(),
        'selected_device': str(config.device),
        'mps_cpu_fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '<unset>'),
        'synthetic_inputs': True,
    }
    print(json.dumps(report, indent=2))
    if config.device.type == 'mps' and report['mps_cpu_fallback'] == '1':
        raise RuntimeError('Unset PYTORCH_ENABLE_MPS_FALLBACK to verify real MPS execution')
    seed_everything(7, config.device)
    linear = torch.nn.Linear(4, 2).to(device=config.device, dtype=torch.float32)
    output = linear(torch.ones(2, 4, device=config.device, dtype=torch.float32))
    output.square().mean().backward()
    assert output.device.type == config.device.type and torch.isfinite(output).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in linear.parameters())

    trainer = ARGUSTrainer(LearnableTrigger(num_tasks=config.num_tasks),
                           ConfidenceEstimator(), OmniModalFusion(), config)
    features = torch.zeros(2, 100, device=config.device, dtype=torch.float32)
    features[:, 82:87] = 0.2
    terms = trainer.compute_loss_terms(features, features.new_full((2, 17), 0.4),
                                       torch.tensor([0, 1], device=config.device))
    terms['total_loss'].backward()
    parameters = {f'{prefix}.{name}': p for prefix, module in
                  [('trigger', trainer.trigger), ('confidence', trainer.confidence), ('fusion', trainer.fusion)]
                  for name, p in module.named_parameters()}
    for name, p in parameters.items():
        assert p.device.type == config.device.type and p.dtype == torch.float32, name
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    if config.device.type == 'mps':
        torch.mps.synchronize()
    elif config.device.type == 'cuda':
        torch.cuda.synchronize()
    print(json.dumps({'linear_forward_backward': 'passed', 'argus_total_backward': 'passed',
                      'actual_device': output.device.type, 'dtype': str(output.dtype),
                      'finite_nonzero_total_gradient_parameters': list(parameters)}, indent=2))


if __name__ == '__main__':
    main()
