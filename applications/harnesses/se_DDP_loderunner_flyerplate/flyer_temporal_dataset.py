import torch
from yoke.datasets.load_npz_dataset import TemporalDataSet


class FlyerTemporalDataSet(TemporalDataSet):
    """Flyerplate-local temporal dataset that forbids zero-timestep pairs."""

    def __getitem__(self, index: int):
        for _ in range(100):
            sample = super().__getitem__(index)
            dt = sample[4]

            if isinstance(dt, torch.Tensor):
                dt_value = float(dt.detach().cpu().item())
            else:
                dt_value = float(dt)

            if dt_value > 0.0:
                return sample

        raise RuntimeError(
            "FlyerTemporalDataSet could not find a valid nonzero-Dt sample."
        )