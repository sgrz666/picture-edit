"""Optional, SHA-bound source-repair records extend existing V5/V6 rows."""
from ..native_v6.data import NativeFullDatasetV6, collate_native_full_v6
from ..native_full_assets import read_record


class NativeFullDatasetV61(NativeFullDatasetV6):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if 'pose_repair_v61' in sample:
            raise ValueError('Repair must be declared as an explicit V6.1 manifest sidecar')
        ref = self.rows[index].get('pose_repair_v61')
        if ref is not None:
            from .spatial import validate_repair_condition
            repair = read_record(ref, schema='native_pose_repair_v61')
            sample['pose_repair_v61'] = validate_repair_condition(repair, sample)
        return sample


def collate_native_full_v61(samples):
    present = ['pose_repair_v61' in sample for sample in samples]
    if any(present):
        if len(samples) != 1 or not all(present):
            raise ValueError('Repair conditions require the supported homogeneous B1 batch')
        from .spatial import validate_repair_condition
        repair = samples[0]['pose_repair_v61']
        batch = collate_native_full_v6([{key: value for key, value in samples[0].items()
                                      if key != 'pose_repair_v61'}])
        batch['pose_repair_v61'] = validate_repair_condition(repair, batch)
        return batch
    return collate_native_full_v6(samples)
