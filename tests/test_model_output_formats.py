import torch

from model import MultiTaskCNN, unpack_model_output


def test_unpacks_modulation_only_tensor_output():
    model = MultiTaskCNN().eval()
    output = model(torch.randn(2, 2, 128))

    mod_logits, snr_pred, anomaly_logit, type_logits = unpack_model_output(output)

    assert mod_logits.shape == (2, 11)
    assert snr_pred is None
    assert anomaly_logit is None
    assert type_logits is None


def test_unpacks_snr_tuple_output():
    model = MultiTaskCNN(use_snr=True).eval()
    output = model(torch.randn(2, 2, 128))

    mod_logits, snr_pred, anomaly_logit, type_logits = unpack_model_output(output)

    assert mod_logits.shape == (2, 11)
    assert snr_pred.shape == (2, 1)
    assert anomaly_logit is None
    assert type_logits is None


def test_unpacks_anomaly_dict_output():
    model = MultiTaskCNN(
        use_snr=True, use_anomaly=True, use_anomaly_type=True
    ).eval()
    output = model(torch.randn(2, 2, 128))

    mod_logits, snr_pred, anomaly_logit, type_logits = unpack_model_output(output)

    assert mod_logits.shape == (2, 11)
    assert snr_pred.shape == (2, 1)
    assert anomaly_logit.shape == (2, 1)
    assert type_logits.shape == (2, 5)
