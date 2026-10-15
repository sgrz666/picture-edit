cd /home/shangguanrz/project/pic-edit/tests
echo ==IMPORTS
grep -h "^from src\|^import src\|from src\." -r . --include="test_*.py" | sed 's/^ *//' | sort | uniq -c | sort -rn | head -40
echo ==TOTAL_TESTS
grep -hc "^def test\|^    def test" -r . --include="test_*.py" | paste -sd+ | bc
echo ==NAMES_SAMPLE
for f in test_adapter_v6_architecture.py test_v65_face_foundation.py test_hi4d_native_binding.py test_qwen21_full_preservation.py test_lab.py test_analysis.py test_report.py test_qwen21_weight_audit.py test_qwen21_condition_combinations.py; do echo "-- $f"; grep "^def test\|^    def test" $f | sed 's/(.*//;s/^ *//' | head -8; done
