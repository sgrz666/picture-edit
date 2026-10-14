$ErrorActionPreference = 'Stop'
$taskLocalRoot = 'D:/codeplus/pictureedit/experiments/bfs_body_face_joint_pilot_20261008'
New-Item -ItemType Directory -Path $taskLocalRoot -Force | Out-Null
$taskExportCode = @'
import tarfile
from pathlib import Path
root = Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_body_face_joint_pilot_20261008')
archive_path = Path('/tmp/bfs_joint_exports_20261008.tar.gz')
with tarfile.open(archive_path, 'w:gz') as archive:
    for folder in ['data', 'preflight', 'evaluation', 'diagnostic_failures', 'quality_reviews', 'report', 'training']:
        for path in (root / folder).rglob('*'):
            if path.is_file() and path.suffix not in ['.pt', '.tmp']:
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
print('Export ready:', round(archive_path.stat().st_size / 1024**2, 1), 'MiB')
'@
$taskExportCode | ssh sg /home/shangguanrz/nihaochen/envs/qwen21/bin/python -
if ($LASTEXITCODE -ne 0) { throw 'Remote export failed' }
scp sg:/tmp/bfs_joint_exports_20261008.tar.gz (Join-Path $taskLocalRoot 'exports.tar.gz')
if ($LASTEXITCODE -ne 0) { throw 'Export download failed' }
tar -xzf (Join-Path $taskLocalRoot 'exports.tar.gz') -C $taskLocalRoot
if ($LASTEXITCODE -ne 0) { throw 'Export extraction failed' }
Write-Output "Images, reports, data locks and metrics saved to $taskLocalRoot"
