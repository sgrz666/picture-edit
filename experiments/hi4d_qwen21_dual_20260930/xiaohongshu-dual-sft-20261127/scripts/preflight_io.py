"""Bounded ordered CPU reads for the full-dataset scans of the Native trainer."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


@contextmanager
def prefetch_dataset(dataset_type, workers):
    """Read ahead during a scan while returning exactly the requested index."""
    get = dataset_type.__getitem__
    windows = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        def prefetched(dataset, index):
            window = windows.setdefault(dataset, {})
            if index not in window:
                for pending in window.values():
                    pending.cancel()
                window.clear()
                window.update({
                    i: executor.submit(get, dataset, i)
                    for i in range(index, min(index + workers, len(dataset)))
                })
            return window.pop(index).result() if window else get(dataset, index)

        dataset_type.__getitem__ = prefetched
        try:
            yield
        finally:
            dataset_type.__getitem__ = get
            for window in windows.values():
                for pending in window.values():
                    pending.cancel()


def trainer_with_prefetch(worker_count, data_report=None):
    from src.pose_control.qwen21_full.native_v62.data import NativeFullDatasetV62
    from src.pose_control.qwen21_full.native_v62.training.io import write_json
    from src.pose_control.qwen21_full.native_v7.training import NativeV7Trainer

    class PrefetchedTrainer(NativeV7Trainer):
        def _data(self):
            with prefetch_dataset(NativeFullDatasetV62, worker_count):
                plan = super()._data()
            if data_report is not None:
                report = plan.report()
                if report.get("passed") is not True:
                    raise RuntimeError("Data preflight rejected")
                write_json(data_report, report)
            return plan

        def _backend(self, data, **options):
            with prefetch_dataset(NativeFullDatasetV62, worker_count):
                return super()._backend(data, **options)

    return PrefetchedTrainer
