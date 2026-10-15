#!/bin/bash
ROOT=/home/shangguanrz/project/pic-edit/experiments
for d in review_20260908 review_20260908_v2 iper_v6_native_20k_v1 iper_v6_comparative_3000 iper_adapter_finetune_v1 iper_adapter_v1 iper_v6_native_finetune_v1 iper_v6_20k_eval_final iper_v6_20k_eval_step4000 iper_v6_testdata_eval deepgen_no_control_ablation_v1 champ_deepgen_formal_v1 iper_test iper_test_1024; do
  echo "===================================================="
  echo "DIR: $d"
  echo "--- realpath ---"
  readlink -f $ROOT/$d
  echo "--- top listing ---"
  ls -la $ROOT/$d
  echo "--- du ---"
  du -sh --si $ROOT/$d 2>/dev/null
  echo "--- file counts by ext (subtree) ---"
  find -L $ROOT/$d -type f 2>/dev/null | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -12
done
