import numpy as np
import torch

from build_dataset import stratified_split
from model import MultiTaskCNN
from train import rotate_phase


def test_modulation_output_shape():
    model = MultiTaskCNN()
    assert model(torch.randn(4, 2, 128)).shape == (4, 11)


def test_parameter_count_is_in_expected_range():
    model = MultiTaskCNN()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 200_000 <= parameter_count <= 260_000


def test_phase_rotation_preserves_unit_power():
    generator = torch.Generator().manual_seed(2026)
    signals = torch.randn(8, 2, 128, generator=generator)
    power = torch.mean(torch.sum(signals ** 2, dim=1), dim=1)
    signals = signals / torch.sqrt(power)[:, None, None]
    angles = torch.rand(8, generator=generator) * (2 * torch.pi)

    rotated = rotate_phase(signals, angles)
    rotated_power = torch.mean(torch.sum(rotated ** 2, dim=1), dim=1)
    assert torch.allclose(rotated_power, torch.ones_like(rotated_power), atol=1e-6)


def test_stratified_split_assigns_each_frame_to_one_split():
    strata = np.repeat(np.arange(4), 20)
    split = stratified_split(strata, np.random.default_rng(2026))
    indices = [set(np.flatnonzero(split == split_id)) for split_id in (0, 1, 2)]

    assert indices[0].isdisjoint(indices[1])
    assert indices[0].isdisjoint(indices[2])
    assert indices[1].isdisjoint(indices[2])
    assert set.union(*indices) == set(range(len(strata)))
    for stratum in np.unique(strata):
        assignments = split[strata == stratum]
        assert set(assignments) == {0, 1, 2}
