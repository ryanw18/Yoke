from pathlib import Path
import numpy as np

from yoke.datasets.load_npz_dataset import TemporalDataSet, LabeledData, import_img_from_npz


class FlyerTemporalDataSet(TemporalDataSet):
    """Flyerplate-local temporal dataset that forbids zero-timestep pairs."""

    def __getitem__(self, index: int):
        index = index % self.n_samples
        depth = int(getattr(self, "_fallback_depth", 0))

        prefix_attempt = 0
        while prefix_attempt < 5:
            file_prefix = self.file_prefix_list[index]
            attempt = 0

            while attempt < self.max_file_checks:
                seq_len = int(
                    self.rng.integers(1, self.max_timeIDX_offset, endpoint=True)
                )
                start_idx = int(self.rng.integers(0, 100 - seq_len, endpoint=True))
                end_idx = start_idx + seq_len

                start_file = f"{file_prefix}_pvi_idx{start_idx:05d}.npz"
                end_file = f"{file_prefix}_pvi_idx{end_idx:05d}.npz"

                start_file_path = Path(self.npz_dir) / file_prefix / start_file
                end_file_path = Path(self.npz_dir) / file_prefix / end_file

                if not (start_file_path.is_file() and end_file_path.is_file()):
                    attempt += 1
                    continue

                try:
                    start_npz = np.load(str(start_file_path), allow_pickle=False)
                except OSError:
                    attempt += 1
                    continue

                try:
                    end_npz = np.load(str(end_file_path), allow_pickle=False)
                except OSError:
                    start_npz.close()
                    attempt += 1
                    continue

                try:
                    ld = LabeledData(
                        str(start_file_path),
                        self.csv_filepath,
                        thermodynamic_variables=self.thermodynamic_variables,
                        kinematic_variables=self.kinematic_variables,
                    )
                    active_npz_field_names = ld.get_active_npz_field_names()
                    active_hydro_field_names = ld.get_active_hydro_field_names()
                    channel_map = ld.get_channel_map()
                    self.all_hydro_field_names = ld.get_all_hydro_field_names()

                    available_start = set(start_npz.files)
                    mask_start = [f in available_start for f in active_npz_field_names]
                    fields_start = [
                        f for f, keep in zip(active_npz_field_names, mask_start) if keep
                    ]
                    chmap_start = [
                        cm for cm, keep in zip(channel_map, mask_start) if keep
                    ]
                    names_start = [
                        nm
                        for nm, keep in zip(active_hydro_field_names, mask_start)
                        if keep
                    ]
                    if not fields_start:
                        start_npz.close()
                        end_npz.close()
                        attempt += 1
                        continue

                    available_end = set(end_npz.files)
                    mask_both = [f in available_end for f in fields_start]

                    present_fields = [
                        f for f, keep in zip(fields_start, mask_both) if keep
                    ]
                    filtered_chmap = [
                        cm for cm, keep in zip(chmap_start, mask_both) if keep
                    ]
                    filtered_names = [
                        nm for nm, keep in zip(names_start, mask_both) if keep
                    ]
                    if not present_fields:
                        start_npz.close()
                        end_npz.close()
                        attempt += 1
                        continue

                    self.active_npz_field_names = present_fields
                    self.channel_map = filtered_chmap
                    self.active_hydro_field_names = filtered_names

                    start_img_list = []
                    end_img_list = []

                    for hfield in present_fields:
                        tmp = import_img_from_npz(start_file_path, hfield)
                        if not self.half_image:
                            tmp = np.concatenate((np.fliplr(tmp), tmp), axis=1)
                        start_img_list.append(tmp)

                        tmp = import_img_from_npz(end_file_path, hfield)
                        if not self.half_image:
                            tmp = np.concatenate((np.fliplr(tmp), tmp), axis=1)
                        end_img_list.append(tmp)

                    start_npz.close()
                    end_npz.close()

                    start_img = np.stack(start_img_list, axis=0)
                    end_img = np.stack(end_img_list, axis=0)
                    dt = np.array([seq_len], dtype=np.float32)

                    return (
                        start_img.astype(np.float32),
                        np.array(self.channel_map, dtype=np.int64),
                        end_img.astype(np.float32),
                        np.array(self.channel_map, dtype=np.int64),
                        dt,
                    )

                except Exception:
                    with np.errstate(all="ignore"):
                        try:
                            start_npz.close()
                        except Exception:
                            pass
                        try:
                            end_npz.close()
                        except Exception:
                            pass
                    attempt += 1
                    continue

            prefix_attempt += 1
            index = (index + 1) % self.n_samples

        if depth > 32:
            raise RuntimeError("FlyerTemporalDataSet could not find a valid nonzero-Dt sample.")

        self._fallback_depth = depth + 1
        try:
            return self.__getitem__((index + 1) % self.n_samples)
        finally:
            self._fallback_depth = depth
