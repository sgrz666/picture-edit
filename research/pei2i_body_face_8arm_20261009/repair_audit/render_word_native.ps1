param([string]$InputDocument,[string]$OutputPdf)
$ErrorActionPreference='Stop'
$wordReportApp=$null
$wordReportDocument=$null
try {
    $wordReportApp=New-Object -ComObject Word.Application
    $wordReportApp.Visible=$false
    $wordReportApp.DisplayAlerts=0
    $wordReportApp.AutomationSecurity=3
    $wordReportDocument=$wordReportApp.Documents.Open($InputDocument,$false,$true,$false)
    $wordReportDocument.Repaginate()
    $wordReportDocument.ExportAsFixedFormat($OutputPdf,17)
    Write-Output ('Rendered pages: '+$wordReportDocument.ComputeStatistics(2))
} finally {
    if ($null -ne $wordReportDocument) {
        $wordReportDocument.Close(0)
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordReportDocument)
    }
    if ($null -ne $wordReportApp) {
        $wordReportApp.Quit()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordReportApp)
    }
}
