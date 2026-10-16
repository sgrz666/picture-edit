import subprocess
import sys


def test_data_package_does_not_load_legacy_training_dependencies():
    result = subprocess.run([sys.executable, "-c", "import sys; import src.data; assert 'src.data.iper_dataset' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
