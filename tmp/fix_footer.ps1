$content = Get-Content 'c:\Users\isley\Desktop\newAnomali\yapayZekaV2.html' -Encoding UTF8 -Raw
$content = $content -replace 'font-size:11px;color:#334155;letter-spacing:3px"', 'font-size:12px;color:#475569;letter-spacing:3px"'
$content = $content -replace 'font-size:9px;color:#1e293b;margin-top:4px"', 'font-size:10px;color:#334155;margin-top:6px"'
Set-Content 'c:\Users\isley\Desktop\newAnomali\yapayZekaV2.html' -Value $content -Encoding UTF8
Write-Host "Done"
