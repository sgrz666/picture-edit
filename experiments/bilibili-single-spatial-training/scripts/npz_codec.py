"""Private-process standard NPZ writer with lossless DEFLATE level 3."""
from contextlib import contextmanager
from pathlib import Path
import zipfile

import numpy as np

COMPRESSLEVEL = 3


def savez_compressed(file, *args, allow_pickle=True, **kwds):
    arrays = dict(kwds)
    for index, value in enumerate(args):
        key = 'arr_' + str(index)
        if key in arrays:
            raise ValueError('Cannot use un-named variables and keyword ' + key)
        arrays[key] = value
    if isinstance(file, (str, bytes, Path)):
        file = str(file)
        if not file.endswith('.npz'):
            file += '.npz'
    with zipfile.ZipFile(file, mode='w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=COMPRESSLEVEL, allowZip64=True) as archive:
        for key, value in arrays.items():
            array = np.asanyarray(value)
            with archive.open(key + '.npy', 'w', force_zip64=True) as stream:
                np.lib.format.write_array(stream, array, allow_pickle=allow_pickle)


@contextmanager
def private_numpy_codec():
    original = np.savez_compressed
    np.savez_compressed = savez_compressed
    try:
        yield
    finally:
        np.savez_compressed = original
