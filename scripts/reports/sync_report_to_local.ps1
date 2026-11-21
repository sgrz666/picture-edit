# 将服务器生成的 workflow-v2 Word 报告同步到本地 D:\codeplus\pictureedit\reports\
$localDir = "D:\codeplus\pictureedit\reports"
if (!(Test-Path $localDir)) {
    New-Item -ItemType Directory -Path $localDir -Force | Out-Null
}
$remotePath = "/home/shangguanrz/project/pic-edit/reports/workflow_v2_两主体交互迁移全流程与实测汇报.docx"
$localPath = Join-Path $localDir "workflow_v2_两主体交互迁移全流程与实测汇报.docx"

Write-Host "正在从远端 sg 下载最新 Word 报告到本地: $localPath" -ForegroundColor Cyan
scp "sg:$remotePath" "$localPath"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 成功下载到本地: $localPath" -ForegroundColor Green
    # 尝试自动在本地调用 Word 打开
    Start-Process "$localPath"
} else {
    Write-Host "❌ 下载失败，请检查 SSH 别名 sg 配置" -ForegroundColor Red
}

