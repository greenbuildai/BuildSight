param (
    [Parameter(Mandatory=$true)]
    [string]$TargetProjectRoot
)

$SourceSkillsPath = "$PSScriptRoot\.agent\skills"
$TargetSkillsPath = Join-Path -Path $TargetProjectRoot -ChildPath ".agent\skills"

# Ensure target .agent/skills directory exists
if (-not (Test-Path $TargetSkillsPath)) {
    Write-Host "Creating directory: $TargetSkillsPath"
    New-Item -ItemType Directory -Force -Path $TargetSkillsPath | Out-Null
}

# Get all skill directories from the source
$Skills = Get-ChildItem -Path $SourceSkillsPath -Directory

foreach ($Skill in $Skills) {
    $LinkPath = Join-Path -Path $TargetSkillsPath -ChildPath $Skill.Name
    $Target = $Skill.FullName

    if (Test-Path $LinkPath) {
        Write-Host "Skill '$($Skill.Name)' already exists in target. Skipping..." -ForegroundColor Yellow
    }
    else {
        Write-Host "Linking skill '$($Skill.Name)' to '$LinkPath'..." -ForegroundColor Green
        # Create Symbolic Link (Requires Admin usually, or slightly relaxed user privs on modern Win10/11)
        # Using Junction for broader compatibility if simple folder link
        cmd /c mklink /J "$LinkPath" "$Target" 
    }
}

Write-Host "Done! Skills from Master are now linked in $TargetProjectRoot" -ForegroundColor Cyan
