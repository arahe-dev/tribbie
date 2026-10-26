param([string]$deck, [string]$outdir)
New-Item -ItemType Directory -Force $outdir | Out-Null
$app = New-Object -ComObject PowerPoint.Application
$p = $app.Presentations.Open($deck, $true, $false, $false)
$i = 1
foreach ($s in $p.Slides) { $s.Export((Join-Path $outdir ("slide{0:D2}.png" -f $i)), "PNG", 1600, 900); $i++ }
$p.Close(); $app.Quit()
"rendered $($i-1)"
