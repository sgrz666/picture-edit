import numpy as np
from src.pose_control.qwen21_full.native_hybrid_assets import detector_peak_response


def test_unbounded_simcc_peaks_are_not_claimed_to_be_probabilities():
    raw=np.array([1.016,.3,.29,-.1,np.nan],np.float32)
    original=raw.copy()
    bounded,record=detector_peak_response(raw)
    np.testing.assert_array_equal(bounded,np.array([1.,.3,.29,0.,0.],np.float32))
    np.testing.assert_array_equal(raw,original)
    np.testing.assert_array_equal(bounded>=.3,np.isfinite(raw)&(raw>=.3))
    assert record['semantics']=='simcc_heatmap_peak_not_calibrated_probability'
    assert record['above_one']==1 and record['nonfinite']==1
