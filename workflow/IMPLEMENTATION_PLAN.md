# Single-person native pose transfer implementation plan

> For agentic workers: execute the independent geometry task with superpowers:subagent-driven-development and apply TDD to geometry, role compilation, and cache validation. Keep all new project files under `workflow/`. Do not commit or change unrelated files in this shared checkout.

Goal: implement and actually run the user's four-stage workflow in `/home/shangguanrz/project/pic-edit/workflow`, using its A/B pictures, no adapter and no LoRA, no basketball.

Architecture: an ordinary Python CLI orchestrates isolated subprocesses using existing geometry and Qwen environments. SMPLest-X exports parameters, camera-space joints and B-space projection. A DWPose-style renderer maps all 134 joints explicitly. PE sees A/B/the final guide; generation receives only A/the final guide with correctly compiled image references. Stage records fingerprint inputs, configuration, model artifacts and outputs before reuse.

Files and responsibilities:

- `run.py`: CLI, automatic single-pair discovery, stage selection, full run, resumability and failure reporting.
- `pose_transfer/common.py`: atomic JSON, file/model provenance, stage fingerprint and output validation, image/canvas preprocessing.
- `pose_transfer/geometry.py`: camera projection, SMPL-X137→DWPose134 semantics, isotropic person-box alignment, drawing and overlays.
- `pose_transfer/prompts.py`: official PE JSON parsing, three-to-two image-role compilation, source-only appearance/objects and generation instruction.
- `smplx_worker.py`: strict A/B person validation, isolated one-frame existing estimator call, SMPL-X reconstruction and projection consistency check.
- `pe_worker.py`: official PE-I2I system prompt and sampling, three inputs, raw/compiled output and timing.
- `generate_worker.py`: native QwenImage21Pipeline, only two images, fixed sampling settings, explicit absent adapter/LoRA record.
- `config.json`, `run.sh`, `README.md`: existing runtime paths, runnable entry, usage and output semantics.
- `tests/test_geometry.py`, `tests/test_prompts.py`, `tests/test_common.py`: real coordinate/role/cache regression checks.

Tasks:

1. [x] Geometry tests first: known pinhole projection, missing/infinite/out-of-frame points, left/right hands with their wrist, face contour reordering, isotropic alignment without changing relative angles. Verify feature-absence failure, implement, and rerun. Source joint mapping must match the installed SMPLest-X human_models.py.
2. [x] Prompt tests first: parse official JSON after thinking, image3→generator image2, reject absent image2 references and require explicit target actions, reject out-of-range/unresolved image references and invalid ratio source; source-only object policy must remain in the generated instruction. Preserve original and compiled prompts separately.
3. [x] Cache tests first: changed source file, configuration, or output content must prevent reuse; atomic output with recorded SHA256 must permit reuse. Create per-run directories and file locks, never reuse just because a PNG exists.
4. [x] Integrate SMPLest-X using a unique one-frame staging directory. Validate one person in each picture. Export parameters and camera metadata, reconstruct vertices/joints using the same neutral SMPL-X model, verify reconstructed mesh and projection against estimator output, then produce native and generation guides plus overlay. Keep all workflow outputs under its run directory and record estimator staging paths inside the run.
5. [x] Integrate official PE with its checkpoint system prompt plus a recorded final-image-role override, `enable_thinking=True`, seed42, temperature1/top_p.95/top_k20. Validate role contract, compile into a two-image instruction, preserve source scene/outfit/objects, exclude target ball, and record actual model inputs. Run in its own subprocess so GPU memory is released before generation.
6. [x] Integrate native Qwen with A+guide only, steps30, trueCFG1, KV cache disabled, seed42. Select a source-aspect canvas within ~1M pixels and store actual references. Refuse any loaded peft config. Save final image, timing and model/reference/config provenance.
7. [x] Deploy only owned new files to sg. Run complete unit suite, CLI help, dependency check, geometry stage, inspect overlay, PE stage, inspect prompt, generation stage, inspect final image. Verify fresh manifest checks all requested artifacts. Rerun identical invocation and confirm all stages are cache hits; mutate a fixture in unit tests to verify stale cache rejection.
8. [x] Document actual commands and actual observed result limitations. Download the output for viewing in this chat and keep the source/test pictures unchanged. Review spec compliance then code quality before final verification.

Test images: A `pictures/A/f2ce3842f4dcda48ddce7e78f4ff5835.jpg` (1280×2196), B `pictures/B/d8898693ef1e7672c739335de57ef44c.jpg` (1294×1279). User explicitly chose source-only objects: basketball is excluded.

