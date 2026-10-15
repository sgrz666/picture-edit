param([string]$HostAlias = 'sg')
$ErrorActionPreference = 'Stop'
$tests = @(
    'tests/test_tiktok_qwen21_full_chain.py',
    'tests/test_tiktok_qwen21_readiness.py',
    'tests/test_tiktok_qwen21_face_cache.py',
    'tests/test_tiktok_qwen21_learning_rates.py',
    'tests/test_tiktok_data_contract.py',
    'tests/test_qwen21_full_bridge.py',
    'tests/test_qwen21_full_preservation.py',
    'tests/test_qwen21_control_contract.py',
    'tests/test_qwen21_dual_control.py',
    'tests/test_qwen21_dual_inference_boundary.py',
    'tests/test_qwen21_face_control.py',
    'tests/test_qwen21_face_loss_alignment.py',
    'tests/test_qwen21_face_objective.py',
    'tests/test_qwen21_face_reference_grid.py',
    'tests/test_qwen21_weight_audit.py',
    'tests/test_qwen21_weight_inheritance.py',
    'tests/test_qwen21_action_prompts.py',
    'tests/test_adapter_v6_architecture.py',
    'tests/test_adapter_reasoning_v1.py'
)
$remote = 'cd /home/shangguanrz/project/pic-edit && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/shangguanrz/nihaochen/envs/qwen21/bin/python -m pytest -q -p no:cacheprovider ' + ($tests -join ' ')
& ssh -o BatchMode=yes -o ClearAllForwardings=yes -o ConnectTimeout=15 $HostAlias $remote 2>&1 |
    Tee-Object -FilePath (Join-Path $PSScriptRoot 'unit_tests_fresh.log')
if ($LASTEXITCODE -ne 0) { throw "Remote tests failed: $LASTEXITCODE" }
