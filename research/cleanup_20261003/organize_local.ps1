$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath 'D:\codeplus\pictureedit').Path.TrimEnd('\')
$auditRoot = Join-Path $taskRoot 'research\cleanup_20261003'
$journal = [System.Collections.Generic.List[object]]::new()

function Assert-WorkspacePath([string] $candidate) {
    $full = [System.IO.Path]::GetFullPath($candidate)
    if (-not $full.StartsWith($taskRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Outside workspace: $full"
    }
    return $full
}

function Save-Journal {
    ConvertTo-Json -InputObject @($journal.ToArray()) -Depth 5 |
        Set-Content -LiteralPath (Join-Path $auditRoot 'local_operations.json') -Encoding utf8
}

function Move-VerifiedTree([string] $relativeSource, [string] $relativeTarget) {
    $source = Assert-WorkspacePath (Join-Path $taskRoot $relativeSource)
    $target = Assert-WorkspacePath (Join-Path $taskRoot $relativeTarget)
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing source: $source" }
    if (Test-Path -LiteralPath $target) { throw "Destination exists: $target" }
    $sourceInfo = Get-Item -LiteralPath $source
    if ($sourceInfo.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Source is link: $source" }
    $manifest = @{}
    foreach ($file in Get-ChildItem -LiteralPath $source -File -Recurse -Force) {
        if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Unexpected link: $($file.FullName)" }
        $manifest[$file.FullName.Substring($source.Length + 1)] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    }
    $item = @{ source = $source; target = $target; hashes = $manifest; state = 'prepared' }
    $journal.Add($item); Save-Journal
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    Move-Item -LiteralPath $source -Destination $target
    foreach ($relative in $manifest.Keys) {
        $actual = (Get-FileHash -LiteralPath (Join-Path $target $relative) -Algorithm SHA256).Hash
        if ($actual -ne $manifest[$relative]) { throw "Hash changed after move: $relative" }
    }
    $item.state = 'completed'; Save-Journal
    Write-Output "MOVED $relativeSource -> $relativeTarget ($($manifest.Count) files)"
}

Move-VerifiedTree 'reportdoc' 'reports'
Move-VerifiedTree 'reports\cleanup_history' 'archive\cleanup_history_202609'
Move-VerifiedTree 'expeiment\tiktok' 'experiments\tiktok_legacy_evidence_202609'

foreach ($relative in @('scripts\polish_figures.py', 'scripts\make_tiktok_overfit_comparisons.py')) {
    $path = Assert-WorkspacePath (Join-Path $taskRoot $relative)
    $original = [IO.File]::ReadAllText($path, [Text.Encoding]::UTF8)
    $before = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
    $backup = Join-Path $auditRoot ('before\' + $relative)
    New-Item -ItemType Directory -Path (Split-Path -Parent $backup) -Force | Out-Null
    Copy-Item -LiteralPath $path -Destination $backup
    $updated = $original.Replace('expeiment\tiktok\one-piece test', 'experiments\tiktok_legacy_evidence_202609\one-piece test')
    if ($updated -eq $original) { throw "Expected reference missing: $relative" }
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $before) { throw "Concurrent edit: $relative" }
    [IO.File]::WriteAllText($path, $updated, [Text.UTF8Encoding]::new($false))
    $journal.Add(@{ source = $path; backup = $backup; before_sha256 = $before; after_sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash; state = 'path_updated' })
    Save-Journal
}

foreach ($relative in @('expeiment', 'tiktok_research\titok-onepiece test', 'visualization\新建文件夹')) {
    $path = Assert-WorkspacePath (Join-Path $taskRoot $relative)
    if (Test-Path -LiteralPath $path) {
        if (@(Get-ChildItem -LiteralPath $path -Force).Count -ne 0) { throw "No longer empty: $path" }
        if ((Get-Item -LiteralPath $path).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Link: $path" }
        Remove-Item -LiteralPath $path
        $journal.Add(@{ source = $path; state = 'empty_directory_removed' }); Save-Journal
    }
}
$indexAfter = (Get-FileHash -LiteralPath (Join-Path $taskRoot '.git\index') -Algorithm SHA256).Hash
$baseline = Get-Content -LiteralPath (Join-Path $auditRoot 'local_before.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ($indexAfter -ne $baseline.index_sha256) { throw 'Git index changed during organization' }
Write-Output 'LOCAL OK: moved tree hashes verified; Git index unchanged.'
