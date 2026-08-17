import torch

from rt_focuser_v2 import build_model, count_parameters


def test_standard_model_and_all_exits():
    model = build_model().eval()
    assert count_parameters(model) == 4_737_490
    image = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        outputs = model.forward_all(image)
    assert set(outputs) == {"exit1", "exit2", "exit3", "exit4", "out"}
    for key in ("exit1", "exit2", "exit3", "exit4"):
        assert outputs[key].shape == image.shape
        assert torch.isfinite(outputs[key]).all()


def test_fixed_exit_selection():
    model = build_model(exit_level=1).eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        for level in range(1, 5):
            model.set_exit_level(level)
            assert model(image).shape == image.shape
