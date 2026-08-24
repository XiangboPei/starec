from types import SimpleNamespace

import torch

from dataset import RecDataset
from model import MODEL_DICT
from model.duorec import DuoRecModel


def make_args(**overrides):
    values = {
        "item_size": 32,
        "hidden_size": 8,
        "max_seq_length": 8,
        "batch_size": 4,
        "initializer_range": 0.02,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_act": "gelu",
        "contrastive_weight": 0.1,
        "contrastive_temperature": 1.0,
        "decomp_level": 2,
        "wave": "db2",
        "alpha": 0.2,
        "use_spectral": 1,
        "use_attention": 1,
        "model_type": "starec",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_release_registry_contains_only_starec_and_duorec():
    assert set(MODEL_DICT) == {"starec", "duorec"}


def test_duorec_dataset_returns_same_target_semantic_positive():
    args = make_args(model_type="duorec", max_seq_length=5)
    dataset = RecDataset(args, [[1, 2, 3, 9, 10], [4, 5, 3, 9, 11]], data_type="train")

    assert len(dataset[0]) == 4
    for index, candidates in enumerate(dataset.semantic_candidates):
        target = dataset.user_seq[index][-1]
        assert all(dataset.user_seq[candidate][-1] == target for candidate in candidates)


def test_duorec_forward_and_loss_are_finite():
    model = DuoRecModel(make_args(model_type="duorec"))
    input_ids = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 3, 4, 5, 6, 7]])
    semantic_ids = torch.tensor([[0, 0, 8, 2, 3, 4, 5, 6], [0, 0, 9, 3, 4, 5, 6, 7]])
    answers = torch.tensor([7, 8])

    output = model(input_ids)
    loss = model.calculate_loss(input_ids, answers, semantic_ids)

    assert output.shape == (2, 8, 8)
    assert torch.isfinite(loss)


def test_starec_forward_and_loss_are_finite():
    model = MODEL_DICT["starec"](make_args())
    input_ids = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6], [0, 0, 0, 3, 4, 5, 6, 7]])
    answers = torch.tensor([7, 8])

    output = model(input_ids)
    loss = model.calculate_loss(input_ids, answers)

    assert output.shape == (2, 8, 8)
    assert torch.isfinite(loss)
