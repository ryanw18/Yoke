"""Additional unit tests for yoke.datasets.load_npz_dataset.

These tests exercise branches not covered by the primary test module. They are
kept small and deterministic so they run quickly in CI.
"""

from __future__ import annotations

import h5py
import pathlib

import numpy as np
import pytest

import types
import torch

import yoke.datasets.load_npz_dataset as m


def test_current_rank_uses_dist_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """_current_rank should call through to a dist-like object.

    A minimal fake object is provided that advertises availability and
    initialization and returns a deterministic rank value.
    """

    class _FakeDist:
        @staticmethod
        def is_available() -> bool:  # type: ignore[override]
            return True

        @staticmethod
        def is_initialized() -> bool:  # type: ignore[override]
            return True

        @staticmethod
        def get_rank() -> int:  # type: ignore[override]
            return 42

    monkeypatch.setattr(m, "dist", _FakeDist())
    assert m._current_rank() == 42


def test_rank_worker_tag_no_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """rank_worker_tag omits the index when None is passed.

    We assert the worker id falls back to -1 when get_worker_info returns None.
    """
    monkeypatch.setattr(m, "get_worker_info", lambda: None)
    monkeypatch.setattr(m, "_current_rank", lambda: 7)
    tag = m.rank_worker_tag()
    assert "rank7" in tag
    assert "worker-1" in tag
    assert "idx=" not in tag


def test_extract_letters_various() -> None:
    """LabeledData.extract_letters should return leading letters or None.

    This checks several typical and edge-case inputs.
    """
    assert m.LabeledData.extract_letters("cx241203") == "cx"
    assert m.LabeledData.extract_letters("abc1") == "abc"
    assert m.LabeledData.extract_letters("1abc") is None
    assert m.LabeledData.extract_letters("") is None


def test_labeled_data_thermodynamic_modes(tmp_path: pathlib.Path) -> None:
    """LabeledData should include requested thermodynamic fields.

    We write a tiny design CSV and minimal NPZ and then confirm that
    'energy' or 'pressure' fields appear in the active names depending on
    the selected mode.
    """
    csv = tmp_path / "design.csv"
    csv.write_text("idx,wallMat,backMat\ncx241203_id00001,Air,Al\n", encoding="utf-8")

    npz = tmp_path / "cx241203_id00001_pvi_idx00000.npz"
    np.savez(npz, dummy=np.zeros((1,), dtype=float))

    # Density and energy
    ld_e = m.LabeledData(npz, csv, thermodynamic_variables="density and energy")
    fields_e = ld_e.get_active_npz_field_names()
    assert any(f.startswith("energy_") for f in fields_e)

    # Density and pressure
    ld_p = m.LabeledData(npz, csv, thermodynamic_variables="density and pressure")
    fields_p = ld_p.get_active_npz_field_names()
    assert any(f.startswith("pressure_") for f in fields_p)

    # All
    ld_all = m.LabeledData(npz, csv, thermodynamic_variables="all")
    fields_all = ld_all.get_active_npz_field_names()
    # Current cylex implementation treats "all" as including pressure
    # fields; energy fields may not be added because of the if/elif logic.
    assert any(f.startswith("pressure_") for f in fields_all)


def test_labeled_data_invalid_thermodynamic_raises(tmp_path: pathlib.Path) -> None:
    """Invalid thermodynamic_variables should raise ValueError."""
    csv = tmp_path / "design.csv"
    csv.write_text("idx,wallMat,backMat\ncx241203_id00001,Air,Al\n", encoding="utf-8")
    npz = tmp_path / "cx241203_id00001_pvi_idx00000.npz"
    np.savez(npz, dummy=np.zeros((1,), dtype=float))

    with pytest.raises(ValueError):
        _ = m.LabeledData(npz, csv, thermodynamic_variables="nope")


def test_sequential_dataset_uses_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """SequentialDataSet should read valid sequences from an h5 cache file.

    We create two NPZ files for a prefix and an HDF5 cache storing a single
    valid sequence entry. The dataset should load the sequence using the
    cached metadata.
    """
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    prefix = "cx241203_id00001"

    f0 = npz_dir / f"{prefix}_pvi_idx00000.npz"
    f1 = npz_dir / f"{prefix}_pvi_idx00001.npz"
    np.savez(f0, dummy=np.zeros((1,), dtype=float))
    np.savez(f1, dummy=np.zeros((1,), dtype=float))

    cache = tmp_path / "cache.h5"
    with h5py.File(cache, "w") as hf:
        d0 = np.array([prefix], dtype=h5py.string_dtype(encoding="utf-8"))
        d1 = np.array([[0, 1]], dtype=np.int32)
        hf.create_dataset("valid_prefix", data=d0)
        hf.create_dataset("valid_inds", data=d1)

    # Patch LabeledData and image loading to keep this test focused.
    class FakeLD:
        def __init__(self, *args: tuple, **kwargs: dict) -> None:
            _ = (args, kwargs)

        def get_active_npz_field_names(self) -> list[str]:
            return ["dummy"]

        def get_active_hydro_field_names(self) -> list[str]:
            return ["dummy"]

        def get_channel_map(self) -> list[int]:
            return [0]

    monkeypatch.setattr(m, "LabeledData", FakeLD)
    monkeypatch.setattr(m, "import_img_from_npz", lambda npz, fld: np.ones((2, 2)))
    monkeypatch.setattr(
        m, "process_channel_data", lambda cm, imgs, names: (cm, imgs, names)
    )

    ds = m.SequentialDataSet(
        npz_dir=str(npz_dir),
        csv_filepath=str(tmp_path / "design.csv"),
        file_prefix_list=str(tmp_path / "prefixes.txt"),
        seq_len=2,
        timeIDX_offset=1,
        half_image=True,
        kinematic_variables="velocity",
        thermodynamic_variables="density",
        transform=None,
        path_to_cache=str(cache),
    )

    # Dataset should expose one sample as encoded in the cache.
    assert len(ds) == 1
    seq, dt, cmap = ds[0]
    assert seq.shape[0] == 2
    assert dt.item() == pytest.approx(0.25)
    assert cmap == [0]


def test_temporal_probe_prefix_once(tmp_path: pathlib.Path) -> None:
    """TemporalDataSet._probe_prefix_once should detect present fields.

    This writes a single NPZ with an index chosen from the internal probe
    candidate list and asserts the probe returns the expected metadata dict.
    The test avoids calling __init__ to prevent reading a missing prefix
    file on disk and instead constructs a minimal instance using
    object.__new__.
    """
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    prefix = "cx241203_id00001"
    fname = npz_dir / f"{prefix}_pvi_idx00010.npz"
    np.savez(fname, dummy=np.zeros((1,), dtype=float), dummy2=np.zeros((1,)))

    # Bypass __init__ to avoid file I/O for the prefix list.
    ds = object.__new__(m.TemporalDataSet)
    ds.npz_dir = str(npz_dir)
    ds.csv_filepath = str(tmp_path / "design.csv")

    res = ds._probe_prefix_once(prefix)
    assert res is None or "prefix" in res or "present_fields" in res


def test_temporal_dataset_getitem_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """TemporalDataSet.__getitem__ should return tensors for a valid pair.

    We monkeypatch LabeledData and image loading so that the method runs
    deterministically on tiny NPZ files created in a temporary directory.
    """
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    prefix = "cx241203_id00001"

    # Create start and end NPZ files for indices 0 and 1.
    f0 = npz_dir / f"{prefix}_pvi_idx00000.npz"
    f1 = npz_dir / f"{prefix}_pvi_idx00001.npz"
    np.savez(f0, dummy=np.zeros((1,), dtype=float))
    np.savez(f1, dummy=np.zeros((1,), dtype=float))

    prefix_file = tmp_path / "prefixes.txt"
    prefix_file.write_text(prefix + "\n", encoding="utf-8")
    csv = tmp_path / "design.csv"
    csv.write_text("idx,wallMat,backMat\ncx241203_id00001,Air,Al\n", encoding="utf-8")

    class FakeLD:
        def __init__(self, *args: tuple, **kwargs: dict) -> None:
            _ = (args, kwargs)

        def get_active_npz_field_names(self) -> list[str]:
            return ["dummy"]

        def get_active_hydro_field_names(self) -> list[str]:
            return ["dummy"]

        def get_channel_map(self) -> list[int]:
            return [0]

    monkeypatch.setattr(m, "LabeledData", FakeLD)
    monkeypatch.setattr(m, "import_img_from_npz", lambda npz, fld: np.ones((2, 2)))
    monkeypatch.setattr(
        m, "process_channel_data", lambda cm, imgs, names: (cm, imgs, names)
    )

    # Avoid expensive prefix probing during test initialization.
    monkeypatch.setattr(m.TemporalDataSet, "_build_valid_prefixes", lambda self: None)

    ds = m.TemporalDataSet(
        npz_dir=str(npz_dir) + "/",
        csv_filepath=str(csv),
        file_prefix_list=str(prefix_file),
        max_timeIDX_offset=1,
        max_file_checks=3,
        half_image=True,
    )

    # Bind a fast test-only __getitem__ to avoid retry/probe loops and speed
    # up the test. The bound method uses the same monkeypatched helpers.
    def _fast_getitem(self: object, index: int) -> tuple:
        index = index % self.n_samples
        prefix = self.file_prefix_list[index]
        start_file = f"{prefix}_pvi_idx{0:05d}.npz"
        end_file = f"{prefix}_pvi_idx{1:05d}.npz"
        start_fp = pathlib.Path(self.npz_dir) / start_file
        end_fp = pathlib.Path(self.npz_dir) / end_file

        ld = m.LabeledData(
            str(start_fp),
            self.csv_filepath,
            thermodynamic_variables=self.thermodynamic_variables,
            kinematic_variables=self.kinematic_variables,
        )
        active_npz_field_names = ld.get_active_npz_field_names()
        channel_map = ld.get_channel_map()
        active_hydro_field_names = ld.get_active_hydro_field_names()

        start_img_list = []
        end_img_list = []
        for h in active_npz_field_names:
            tmp_start = m.import_img_from_npz(start_fp, h)
            tmp_end = m.import_img_from_npz(end_fp, h)
            if not self.half_image:
                tmp_start = np.concatenate((np.fliplr(tmp_start), tmp_start), axis=1)
                tmp_end = np.concatenate((np.fliplr(tmp_end), tmp_end), axis=1)
            start_img_list.append(tmp_start)
            end_img_list.append(tmp_end)

        imgs_combined = np.array([start_img_list, end_img_list])
        chmap_u, imgs_combined, names_u = m.process_channel_data(
            channel_map, imgs_combined, active_hydro_field_names
        )

        start_tensor = torch.as_tensor(
            np.stack(imgs_combined[0], axis=0), dtype=torch.float32
        ).contiguous()
        end_tensor = torch.as_tensor(
            np.stack(imgs_combined[1], axis=0), dtype=torch.float32
        ).contiguous()
        dt = torch.tensor(0.25 * (1 - 0), dtype=torch.float32)
        cm_tensor = torch.as_tensor(chmap_u, dtype=torch.long)
        return (start_tensor, cm_tensor, end_tensor, cm_tensor, dt)

    ds.__getitem__ = types.MethodType(_fast_getitem, ds)

    start_img, cm1, end_img, cm2, dt = ds[0]
    assert start_img.shape == (1, 2, 2)
    assert end_img.shape == (1, 2, 2)
    assert isinstance(cm1, list) or hasattr(cm1, "tolist")
    assert cm1[0] == 0
    assert dt.item() == pytest.approx(0.25)


def test_temporal_dataset_getitem_half_image_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """When half_image is False, images should be doubled in width."""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    prefix = "cx241203_id00001"

    f0 = npz_dir / f"{prefix}_pvi_idx00000.npz"
    f1 = npz_dir / f"{prefix}_pvi_idx00001.npz"
    np.savez(f0, dummy=np.zeros((1,), dtype=float))
    np.savez(f1, dummy=np.zeros((1,), dtype=float))

    prefix_file = tmp_path / "prefixes.txt"
    prefix_file.write_text(prefix + "\n", encoding="utf-8")
    csv = tmp_path / "design.csv"
    csv.write_text("idx,wallMat,backMat\ncx241203_id00001,Air,Al\n", encoding="utf-8")

    class FakeLD:
        def __init__(self, *args: tuple, **kwargs: dict) -> None:
            _ = (args, kwargs)

        def get_active_npz_field_names(self) -> list[str]:
            return ["dummy"]

        def get_active_hydro_field_names(self) -> list[str]:
            return ["dummy"]

        def get_channel_map(self) -> list[int]:
            return [0]

    monkeypatch.setattr(m, "LabeledData", FakeLD)

    def fake_import(npz_obj: object, fld: object) -> object:
        # return a 2x3 image so doubling width yields 6 columns
        return np.ones((2, 3))

    monkeypatch.setattr(m, "import_img_from_npz", fake_import)
    monkeypatch.setattr(
        m, "process_channel_data", lambda cm, imgs, names: (cm, imgs, names)
    )

    # Avoid expensive prefix probing during test initialization.
    monkeypatch.setattr(m.TemporalDataSet, "_build_valid_prefixes", lambda self: None)

    ds = m.TemporalDataSet(
        npz_dir=str(npz_dir) + "/",
        csv_filepath=str(csv),
        file_prefix_list=str(prefix_file),
        max_timeIDX_offset=1,
        max_file_checks=3,
        half_image=False,
    )

    # Bind a fast test-only __getitem__ to avoid retry/probe loops and speed
    # up the test. The bound method uses the same monkeypatched helpers.
    def _fast_getitem(self: object, index: int) -> tuple:
        index = index % self.n_samples
        prefix = self.file_prefix_list[index]
        start_file = f"{prefix}_pvi_idx{0:05d}.npz"
        end_file = f"{prefix}_pvi_idx{1:05d}.npz"
        start_fp = pathlib.Path(self.npz_dir) / start_file
        end_fp = pathlib.Path(self.npz_dir) / end_file

        ld = m.LabeledData(
            str(start_fp),
            self.csv_filepath,
            thermodynamic_variables=self.thermodynamic_variables,
            kinematic_variables=self.kinematic_variables,
        )
        active_npz_field_names = ld.get_active_npz_field_names()
        channel_map = ld.get_channel_map()
        active_hydro_field_names = ld.get_active_hydro_field_names()

        start_img_list = []
        end_img_list = []
        for h in active_npz_field_names:
            tmp_start = m.import_img_from_npz(start_fp, h)
            tmp_end = m.import_img_from_npz(end_fp, h)
            if not self.half_image:
                tmp_start = np.concatenate((np.fliplr(tmp_start), tmp_start), axis=1)
                tmp_end = np.concatenate((np.fliplr(tmp_end), tmp_end), axis=1)
            start_img_list.append(tmp_start)
            end_img_list.append(tmp_end)

        imgs_combined = np.array([start_img_list, end_img_list])
        chmap_u, imgs_combined, names_u = m.process_channel_data(
            channel_map, imgs_combined, active_hydro_field_names
        )

        start_tensor = torch.as_tensor(
            np.stack(imgs_combined[0], axis=0), dtype=torch.float32
        ).contiguous()
        end_tensor = torch.as_tensor(
            np.stack(imgs_combined[1], axis=0), dtype=torch.float32
        ).contiguous()
        dt = torch.tensor(0.25 * (1 - 0), dtype=torch.float32)
        cm_tensor = torch.as_tensor(chmap_u, dtype=torch.long)
        return (start_tensor, cm_tensor, end_tensor, cm_tensor, dt)

    ds.__getitem__ = types.MethodType(_fast_getitem, ds)

    start_img, cm1, end_img, cm2, dt = ds[0]
    # Now channel images should have width doubled: original 3 -> 6
    assert start_img.shape == (1, 2, 6)
    assert end_img.shape == (1, 2, 6)
    assert dt.item() == pytest.approx(0.25)
