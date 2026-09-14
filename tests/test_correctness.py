"""Issue #3 correctness regressions. All inputs are synthetic, not performance data."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from argus import ConfidenceEstimator, LearnableTrigger, OmniModalFusion
from argus.inference import ARGUSDetector
from argus.train import PhishingSampleDataset
from argus.training.trainer import ARGUSTrainer
from argus.utils.config import Config


def make_trainer():
    torch.manual_seed(7)
    config = Config()
    config.device = torch.device('cpu')
    return ARGUSTrainer(LearnableTrigger(num_tasks=config.num_tasks),
                        ConfidenceEstimator(), OmniModalFusion(), config)


def synthetic_batch(labels=(0, 1)):
    features = torch.zeros(len(labels), 100)
    features[:, 82:87] = 0.2
    return {'features': features, 'prior_scores': torch.full((len(labels), 17), 0.4),
            'label': torch.tensor(labels)}


def test_loss_uses_probabilities():
    trainer = make_trainer()
    batch = synthetic_batch()
    probabilities = torch.tensor([0.2, 0.8])
    risks = {m: probabilities for m in trainer.fusion.modality_names}
    _, details = trainer.compute_loss(batch['features'], batch['prior_scores'],
                                      batch['label'], risk_scores=risks)
    expected = torch.nn.functional.binary_cross_entropy(probabilities, batch['label'].float())
    assert details['detection_loss'] == pytest.approx(expected.item())


def test_fusion_weights_receive_detection_gradients():
    fusion = OmniModalFusion()
    risks = {m: torch.tensor([p], requires_grad=True) for m, p in
             zip(fusion.modality_names, [0.1, 0.4, 0.9])}
    probability = fusion(risks, {m: torch.ones(1) for m in risks})
    torch.nn.functional.binary_cross_entropy(probability, torch.ones(1)).backward()
    for parameter in fusion.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize('fallback', [False, True])
def test_fusion_preserves_float64(fallback):
    fusion = OmniModalFusion().double()
    risks = {'quantitative': torch.tensor([0.2, 0.8], dtype=torch.float64)}
    confidence = torch.zeros(2, dtype=torch.float64) if fallback else torch.ones(2, dtype=torch.float64)
    if fallback:
        with pytest.warns(UserWarning, match='No available modalities'):
            probability, _ = fusion.fuse_with_confidence(risks, {'quantitative': confidence})
    else:
        probability, _ = fusion.fuse_with_confidence(risks, {'quantitative': confidence})
    assert probability.dtype == torch.float64


class FixedProbabilityFusion(OmniModalFusion):
    def forward(self, risk_scores, confidences, **kwargs):
        return torch.full_like(risk_scores['quantitative'], 0.2)


def test_evaluation_does_not_sigmoid_probabilities_again():
    trainer = make_trainer()
    trainer.fusion = FixedProbabilityFusion()
    # Compare classification, not a claimed real dataset metric.
    report = trainer.evaluate([synthetic_batch()])
    assert report['recall'] == 0.0


@pytest.mark.parametrize('labels', [(1, 1), (0, 0), ()])
def test_evaluation_rejects_invalid_binary_split(labels):
    loader = [synthetic_batch(labels)] if labels else []
    with pytest.raises(ValueError, match='class|empty'):
        make_trainer().evaluate(loader)


def make_dataset(tmp_path, metadata, extractor=None, prior=None, **kwargs):
    folder = tmp_path / 'sample'
    folder.mkdir(exist_ok=True)
    (folder / 'info.txt').write_text(metadata)
    (folder / 'html.txt').write_text('<html>synthetic fixture</html>')
    extractor = extractor or SimpleNamespace(extract_features=lambda *a: np.zeros(100, dtype=np.float32))
    prior = prior or SimpleNamespace(task_names=['task'], compute_trigger_scores=lambda x: torch.zeros(1, 1))
    return PhishingSampleDataset(str(tmp_path), extractor, prior, train_ratio=1.0, **kwargs)


def test_label_is_not_inferred_from_url_substrings(tmp_path):
    with pytest.raises(ValueError, match='label'):
        make_dataset(tmp_path, 'url: https://legitimate-phishing.example/\n')


@pytest.mark.parametrize('stage', ['feature', 'prior'])
def test_failed_extraction_is_not_a_zero_observation(tmp_path, stage):
    def fail(*args):
        raise RuntimeError('synthetic extraction failure')
    kwargs = {'extractor': SimpleNamespace(extract_features=fail)} if stage == 'feature' else {
        'prior': SimpleNamespace(task_names=['task'], compute_trigger_scores=fail)}
    with pytest.warns(UserWarning, match='single-class'):
        dataset = make_dataset(tmp_path, 'url: https://example.com\nlabel: phishing\n', **kwargs)
    item = dataset[0]
    assert item.get('error_stage') == stage
    assert 'features' not in item and 'prior_scores' not in item


def synthetic_detector(monkeypatch):
    monkeypatch.setattr(Config, 'device', torch.device('cpu'))
    detector = ARGUSDetector()
    monkeypatch.setattr(detector, 'extract_features', lambda *a: torch.zeros(100))
    monkeypatch.setattr(detector, 'trigger_tasks', lambda *a: {'triggered_tasks': ['typosquatting_detection']})
    monkeypatch.setattr(detector, 'compute_confidences', lambda *a: {
        'quantitative': torch.tensor(0.9), 'visual': torch.tensor(0.8), 'semantic': torch.tensor(0.95)})
    monkeypatch.setattr(detector.conflict_detector, 'detect_conflicts', lambda *a: {
        'brand_domain_mismatch': True, 'visual_semantic_mismatch': False})
    return detector


def test_inference_uncertainty_uses_actual_confidences(monkeypatch):
    result = synthetic_detector(monkeypatch).detect('https://example.com')
    assert result['uncertainty'] == pytest.approx(0.2)


def test_inference_reports_only_true_conflicts(monkeypatch):
    result = synthetic_detector(monkeypatch).detect('https://example.com')
    assert result['conflicts_detected'] == ['brand_domain_mismatch']


@pytest.mark.parametrize('stage', ['url', 'html', 'whois'])
def test_internal_extraction_errors_are_surfaced(tmp_path, monkeypatch, stage):
    import argus.core.feature_extractor as module
    extractor = module.PhishingFeatureExtractor()
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic internal extraction failure')
    monkeypatch.setattr(module, 'WHOIS_AVAILABLE', False)
    if stage == 'url':
        monkeypatch.setattr(extractor, '_calculate_entropy', fail)
    elif stage == 'html':
        monkeypatch.setattr(module, 'BS4_AVAILABLE', True)
        monkeypatch.setattr(module, 'BeautifulSoup', fail, raising=False)
    else:
        monkeypatch.setattr(module, 'WHOIS_AVAILABLE', True)
        monkeypatch.setattr(module, 'whois', SimpleNamespace(whois=fail), raising=False)
    with pytest.warns(UserWarning, match='single-class'):
        dataset = make_dataset(tmp_path, 'url: https://example.com\nlabel: 1\n', extractor=extractor)
    assert dataset[0].get('error_stage') == 'feature'


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA hardware unavailable'))])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('case', ['all', 'missing', 'scalar', 'reshape', 'fallback'])
def test_fusion_device_dtype_and_backward(device, dtype, case):
    fusion = OmniModalFusion().to(device=device, dtype=dtype).eval()
    shape = () if case == 'scalar' else (2, 1) if case == 'reshape' else (2,)
    risks = {m: torch.full(shape, p, device=device, dtype=dtype, requires_grad=True)
             for m, p in zip(fusion.modality_names, [0.1, 0.4, 0.9])}
    confs = {m: torch.full(shape, 0.0 if case == 'fallback' else 0.8, device=device, dtype=dtype)
             for m in risks}
    if case == 'missing':
        del risks['visual']
        del confs['visual']
    if case == 'fallback':
        with pytest.warns(UserWarning, match='No available modalities'):
            probabilities, _ = fusion.fuse_with_confidence(risks, confs)
    else:
        probabilities, _ = fusion.fuse_with_confidence(risks, confs)
        probabilities.sum().backward()
        for risk in risks.values():
            assert risk.grad is not None and risk.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in fusion.parameters())
    assert probabilities.device.type == device
    assert probabilities.dtype == dtype
    assert torch.isfinite(probabilities).all()


def test_empty_fusion_uses_module_dtype():
    fusion = OmniModalFusion().double()
    with pytest.warns(UserWarning, match='No available modalities'):
        probability, info = fusion.fuse_with_confidence({}, {})
    assert probability.dtype == torch.float64
    assert info['error'] == 'no_available_modalities'


def test_scalar_defaults_and_adjusted_weights_preserve_gradient():
    fusion = OmniModalFusion().double()
    adjusted = {m: torch.tensor(1 / 3, dtype=torch.float64, requires_grad=True) for m in fusion.modality_names}
    probability, _ = fusion.fuse_with_confidence({'quantitative': 0.1, 'visual': 0.4, 'semantic': 0.9},
                                                {}, adjusted_weights=adjusted)
    probability.sum().backward()
    assert probability.dtype == torch.float64
    assert all(w.grad is not None and w.grad.abs() > 0 for w in adjusted.values())
    assert all(p.grad is None for p in fusion.parameters())  # Explicit override, as before.


def test_probability_loss_boundary_is_finite():
    trainer = make_trainer()
    scores = torch.tensor([0.0, 1.0], requires_grad=True)
    loss = trainer.detection_criterion(scores, torch.tensor([1.0, 0.0]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(scores.grad).all()


def test_uncertainty_source_handles_batched_missing_confidence():
    estimator = ConfidenceEstimator().double()
    result = estimator.compute_overall_confidence({'quantitative': torch.tensor([0.2, 0.9], dtype=torch.float64)})
    torch.testing.assert_close(result['uncertainty'], torch.tensor([0.8, 0.2], dtype=torch.float64))


def test_inference_llm_and_image_remain_explicit_stubs(monkeypatch):
    detector = synthetic_detector(monkeypatch)
    result = detector.detect('https://example.com', image='/nonexistent-unused.png', use_llm=True)
    assert result['llm_status'] == 'prompt_only_stub'
    assert result['visual_status'] == 'placeholder_image_unused'
    assert result['semantic_status'] == 'placeholder'
    assert result['prompt'] and result['llm_response'] is None


@pytest.mark.parametrize('text', ['label: 2', 'label: unknown', 'label: 0\nclass: 1'])
def test_invalid_or_conflicting_labels_fail(tmp_path, text):
    with pytest.raises(ValueError, match='label'):
        make_dataset(tmp_path, 'url: https://example.com\n' + text, default_label=1)


def test_explicit_default_label_policy_is_counted(tmp_path):
    with pytest.warns(UserWarning, match='single-class'):
        dataset = make_dataset(tmp_path, 'url: https://example.com', default_label=1)
    assert dataset.label_counts == {0: 0, 1: 1}
    assert dataset.defaulted_label_count == 1
    assert dataset[0]['label'].item() == 1


def test_missing_url_is_not_fabricated(tmp_path):
    with pytest.raises(ValueError, match='URL'):
        make_dataset(tmp_path, 'label: 1')


class WorkerExtractor:
    def extract_features(self, url, html):
        if 'fail' in url:
            raise RuntimeError('synthetic worker failure')
        return np.zeros(100, dtype=np.float32)


class WorkerPrior:
    task_names = Config.TASK_NAMES
    def compute_trigger_scores(self, features):
        return torch.zeros(features.shape[0], len(self.task_names))


@pytest.mark.parametrize('workers', [0, 2])
def test_worker_failures_are_counted_and_evaluation_aborts(tmp_path, workers):
    from argus.training.data_validation import collate_samples
    for i in range(4):
        folder = tmp_path / str(i)
        folder.mkdir()
        (folder / 'info.txt').write_text(f'url: https://{"fail" if i == 0 else "ok"}.example/{i}\nlabel: {i % 2}')
    dataset = PhishingSampleDataset(str(tmp_path), WorkerExtractor(), WorkerPrior(), train_ratio=1.0)
    loader = DataLoader(dataset, batch_size=4, num_workers=workers, collate_fn=collate_samples)
    trainer = make_trainer()
    with pytest.raises(ValueError, match='Extraction failures'):
        trainer.evaluate(loader)
    assert trainer.data_report['attempted_samples'] == 4
    assert trainer.data_report['failure_counts'] == {'feature': 1, 'prior': 0, 'files': 0}
    assert trainer.history['val_accuracy'] == []


@pytest.mark.parametrize('stage', ['feature', 'prior', 'files'])
def test_all_failed_batches_abort_before_optimization(stage):
    from argus.training.data_validation import collate_samples
    trainer = make_trainer()
    before = {name: p.detach().clone() for name, p in trainer.trigger.named_parameters()}
    batch = collate_samples([{'folder_name': 'synthetic', 'error_stage': stage, 'error': 'test'}])
    with pytest.raises(ValueError, match='Extraction failures'):
        trainer.train_epoch([batch], 0)
    assert trainer.data_report['failure_counts'][stage] == 1
    assert trainer.history['train_loss'] == []
    for name, parameter in trainer.trigger.named_parameters():
        torch.testing.assert_close(parameter, before[name])


@pytest.mark.parametrize('stage', ['feature', 'prior'])
def test_nonfinite_extraction_is_a_countable_failure(tmp_path, stage):
    kwargs = {'extractor': SimpleNamespace(extract_features=lambda *a: np.full(100, np.nan, dtype=np.float32))} if stage == 'feature' else {
        'prior': SimpleNamespace(task_names=['task'], compute_trigger_scores=lambda x: torch.full((1, 1), float('inf')))}
    with pytest.warns(UserWarning, match='single-class'):
        dataset = make_dataset(tmp_path, 'url: https://example.com\nlabel: 1', **kwargs)
    assert dataset[0]['error_stage'] == stage


@pytest.mark.parametrize('split', ['val', 'test'])
def test_dataset_preflight_refuses_single_class_evaluation(tmp_path, split):
    folder = tmp_path / 'sample'
    folder.mkdir()
    (folder / 'info.txt').write_text('url: https://example.com\nlabel: 1')
    with pytest.raises(ValueError, match='single-class'):
        PhishingSampleDataset(str(tmp_path), WorkerExtractor(), WorkerPrior(), split=split,
                              train_ratio=0, val_ratio=1 if split == 'val' else 0)


def test_evaluation_validates_labels_and_counts_before_metrics():
    trainer = make_trainer()
    report = trainer.evaluate([synthetic_batch()])
    assert report['data_report']['label_counts'] == {0: 1, 1: 1}
    assert report['data_report']['attempted_samples'] == 2
    assert not any(report['data_report']['failure_counts'].values())
    with pytest.raises(ValueError, match='binary labels'):
        trainer.evaluate([synthetic_batch((0, 2))])
    with pytest.raises(ValueError, match='empty'):
        trainer.evaluate([synthetic_batch(())])


def test_state_dict_keys_shapes_and_checkpoint_roundtrip(tmp_path):
    trainer = make_trainer()
    expected = {
        'trigger': {'alpha_logit': (), 'residual_net.0.weight': (64, 100), 'residual_net.0.bias': (64,),
                    'residual_net.3.weight': (64, 64), 'residual_net.3.bias': (64,),
                    'residual_net.6.weight': (17, 64), 'residual_net.6.bias': (17,)},
        'confidence': {name: () for name in ['trustworthiness_weight', 'coverage_weight', 'consistency_weight']},
        'fusion': {f'weight_logits.{name}': () for name in ['quantitative', 'visual', 'semantic']},
    }
    for name, shapes in expected.items():
        assert {key: tuple(value.shape) for key, value in getattr(trainer, name).state_dict().items()} == shapes
    checkpoint = str(tmp_path / 'synthetic-state.pt')
    trainer.save_checkpoint(checkpoint, 0, {})
    restored = make_trainer()
    assert restored.load_checkpoint(checkpoint) == 0
    for name in expected:
        for key, value in getattr(trainer, name).state_dict().items():
            torch.testing.assert_close(getattr(restored, name).state_dict()[key], value)
