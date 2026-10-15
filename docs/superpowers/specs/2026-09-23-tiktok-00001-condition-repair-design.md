# TikTok 00001 condition repair design

## Scope

Repair only the `0014 -> 0074` evaluation pair. Preserve the existing 127-frame
`v6_conditions.pt`, raw images, masks, and earlier experiment outputs. Put every
derived condition and image in a new experiment directory. No DWPose, HaMeR, or
other new estimator is required for this experiment.

## Data flow

SMPLest-X `joint_proj_model[frame, :25]` uses the named 25-body-joint order in
`add/tools/SMPLest-X/human_models/human_models.py`. Convert its 12x16 heatmap
coordinates back to the 604x1080 RGB frame with the saved focal/principal-point
crop transform already used for the face 72 landmarks. Reorder to OpenPose-25,
mark out-of-frame points invalid, then render the current 25 heatmaps. Fail if
the projected joints or camera metadata are missing; never substitute mesh
vertices.

For the second ablation, use existing SMPL-X mesh and skinning weights with
nvdiffrast to rasterize one 14-class part map. This is a renderer, not another
pose model. A pose-only result is still useful if nvdiffrast cannot be installed.

## Evaluation

Use the same source image, target frame, checkpoint, seed, CFG, and 28 denoising
steps for baseline, corrected-pose, and corrected-pose-plus-part. Compare Face
OFF before Face ON. Review the 25-point overlay and part-map alignment before
running inference. The existing one-step low-noise preview is not a generative
success criterion.

After condition validation, train one TikTok face overfit run for 1500 steps,
with the approved 1000-step head/gate warmup, using only the corrected sample.
Keep its checkpoint and outputs separate from the earlier 200-step run.
