"""Resume completed estimator substages without rewriting protected outputs."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def test_completed_substage_is_reused_and_changed_output_is_rejected(tmp_path):
    from abc_geometry_worker import checkpointed_substage
    output=tmp_path/'mesh.npz';marker=tmp_path/'completed.json';calls=[]
    def work():calls.append(1);output.write_bytes(b'validated mesh')
    assert checkpointed_substage(marker,'input-sha',work,lambda:[output]) is False
    before=output.stat().st_mtime_ns
    assert checkpointed_substage(marker,'input-sha',lambda:pytest.fail('estimator was repeated'),lambda:[output]) is True
    assert len(calls)==1 and output.stat().st_mtime_ns==before
    output.write_bytes(b'changed mesh')
    with pytest.raises(RuntimeError,match='evidence'):
        checkpointed_substage(marker,'input-sha',lambda:pytest.fail('overwriting output'),lambda:[output])
