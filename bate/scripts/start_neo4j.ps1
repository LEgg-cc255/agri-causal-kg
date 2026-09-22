# ============================================================
# 研究内容2 Day 1: Neo4j 启动脚本 (支持 2 种路径)
# ------------------------------------------------------------
# 用法 (PowerShell):
#   1) cd 到 该脚本所在目录 bate/
#   2) 设密码（首次需要；该密码与研究内容2实验绑定: paper2026）
#        $env:NEO4J_PASSWORD = "paper2026"
#   3a) 方案 A (推荐): 使用 Docker 启动（需先装 Docker Desktop + WSL2/HyperV）
#        .\scripts\start_neo4j.ps1 docker
#   3b) 方案 B: 使用 Neo4j Desktop/本地已安装版
#        # 在 Neo4j Desktop 里手动启动项目后，
#        .\scripts\start_neo4j.ps1 check
#   4) 导入数据
#        python -m bate.scripts.import_to_neo4j
# ============================================================

param(
    [Parameter(Position = 0)]
    [ValidateSet("docker", "local", "check", "stop", "logs")]
    [string]$Mode = "check",
    [string]$Neo4jHome = $env:NEO4J_HOME,
    [string]$JavaHome = $env:JAVA_HOME
)

$ContainerName = "bate-neo4j"
$Image       = "neo4j:5"
$PW          = $env:NEO4J_PASSWORD
$DataDir     = Resolve-Path (Join-Path $PSScriptRoot "..\data\neo4j_data") -ErrorAction SilentlyContinue
if (-not $DataDir) {
    $DataDir = Join-Path $PSScriptRoot "..\data\neo4j_data"
    New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
    $DataDir = Resolve-Path $DataDir
}
$BoltPort  = 7687
$HttpPort  = 7474
$HttpsPort = 7473

function Write-Step([string]$msg) { Write-Host "[*] $msg" -ForegroundColor Cyan }
function Write-OK  ([string]$msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Fail([string]$msg) { Write-Host "[X] $msg" -ForegroundColor Red }

# --------- 前置条件: 密码 ----------
# 账号固定为 neo4j（Neo4j 默认）；密码用户指定为 123456
if (-not $PW) {
    Write-Warn "Environment NEO4J_PASSWORD not set. Using user-specified default: 123456"
    Write-Warn "  (Override anytime:  `$env:NEO4J_PASSWORD = '<your-password>')"
    $PW = "123456"
    $env:NEO4J_PASSWORD = $PW
}

switch ($Mode) {
# ==============================================================
#  local   : Run Neo4j Community Edition (ZIP install)
#            Usage: .\scripts\start_neo4j.ps1 local -Neo4jHome "C:\neo4j\neo4j-community-5.x"
# ==============================================================
"local" {
    if (-not $Neo4jHome -or -not (Test-Path (Join-Path $Neo4jHome "bin\neo4j.bat"))) {
        Write-Fail "Neo4jHome not found or invalid. Pass -Neo4jHome 'C:\path\to\neo4j-community-5.x' (or set env NEO4J_HOME)."
        Write-Host "  Example: .\scripts\start_neo4j.ps1 local -Neo4jHome 'C:\neo4j\neo4j-community-5.26.0'"
        exit 1
    }
    $Neo4jBat = Join-Path $Neo4jHome "bin\neo4j.bat"

    # JAVA_HOME: 必须 JDK 17+（Neo4j 5.x 硬性要求）。用户机器是 JDK8，需显式指定 17
    if (-not $JavaHome) {
        $cand = @(
            "C:\Program Files\Eclipse Adoptium\jdk-17*",
            "C:\Program Files\Microsoft\jdk-17*",
            "C:\Program Files\Zulu\jdk-17*",
            "C:\Program Files\BellSoft\LibericaJDK-17*"
        )
        foreach ($c in $cand) {
            $hit = Get-ChildItem -Path $c -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($hit) { $JavaHome = $hit.FullName; break }
        }
    }
    if (-not $JavaHome) {
        Write-Fail "JDK 17 not found. Neo4j 5.x requires JDK 17+ (current JAVA_HOME is JDK8)."
        Write-Host "  Install: https://adoptium.net/temurin/releases/?version=17  (Temurin 17 MSI, x64)"
        Write-Host "  Then rerun: .\scripts\start_neo4j.ps1 local -Neo4jHome '...' -JavaHome 'C:\Program Files\Eclipse Adoptium\jdk-17.x'"
        exit 2
    }
    Write-Step "Using JAVA_HOME=$JavaHome"
    $env:JAVA_HOME = $JavaHome
    $env:JAVA_HOME_OVERRIDE = $JavaHome  # neo4j.bat 优先认这个

    # 首次启动会自动用 neo4j/neo4j 默认密码；若已有 data dir 则跳过
    Write-Step "Starting Neo4j Community (console) at $Neo4jHome ..."
    Write-Host "  (Ctrl+C 停止。若想后台运行: neo4j start)"
    Write-Host ""
    Push-Location $Neo4jHome
    & $Neo4jBat console
    Pop-Location
}

# ==============================================================
#  docker  : Run Neo4j 5.x in Docker (preferred)
# ==============================================================
"docker" {
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $docker) {
        Write-Fail "docker CLI not found. Install Docker Desktop first:  https://docs.docker.com/desktop/install/windows-install/"
        Write-Host "  (During install, select 'Use WSL2 instead of Hyper-V' if WSL2 enabled on your machine)"
        exit 1
    }
    try {
        $null = & docker version --format '{{.Server.Version}}' 2>$null
    } catch {
        Write-Fail "Docker Engine not running. Start Docker Desktop from Start Menu, wait 1-2 min until the whale icon shows stable (no-animation), then rerun."
        exit 2
    }
    Write-Step "Docker Engine OK"

    # Stop existing old container
    $running = & docker ps -a --format '{{.Names}}' | Where-Object { $_ -eq $ContainerName }
    if ($running) {
        $status = & docker inspect --format '{{.State.Status}}' $ContainerName 2>$null
        if ($status -eq "running") {
            Write-OK "Container '$ContainerName' is already running (status=$status)."
        } else {
            Write-Step "Start existing container '$ContainerName'..."
            & docker start $ContainerName | Out-Null
        }
    } else {
        Write-Step "Pulling neo4j:5 image (only on first run, ~500MB)... then start container."
        $Auth = "neo4j/$PW"
        & docker run -d --name $ContainerName `
            -p ${BoltPort}:7687 -p ${HttpPort}:7474 -p ${HttpsPort}:7473 `
            -v "${DataDir}\data:/data" `
            -v "${DataDir}\logs:/logs" `
            -v "${DataDir}\import:/import" `
            -v "${DataDir}\plugins:/plugins" `
            -e "NEO4J_AUTH=${Auth}" `
            -e "NEO4J_ACCEPT_LICENSE_AGREEMENT=yes" `
            -e "NEO4J_server_memory_heap_initial__size=1G" `
            -e "NEO4J_server_memory_heap_max__size=2G" `
            $Image | Out-Null
    }
    # Wait for ready
    Write-Step "Waiting for Neo4j Bolt port (${BoltPort})..."
    $retry = 30
    while ($retry-- -gt 0) {
        try {
            $c = New-Object System.Net.Sockets.TcpClient
            $c.Connect("127.0.0.1", $BoltPort)
            $c.Close()
            break
        } catch { Start-Sleep -Seconds 2 }
    }
    if ($retry -le 0) {
        Write-Fail "Timeout waiting for Bolt port 7687. Inspect: docker logs bate-neo4j"
        exit 3
    }
    Write-OK  "Neo4j is UP."
    Write-Host ""
    Write-Host "  Neo4j Browser (visual UI):  http://localhost:${HttpPort}"
    Write-Host "  Bolt URI for client:         bolt://localhost:${BoltPort}"
    Write-Host "  Username:                    neo4j"
    Write-Host "  Password:                    (same as `$env:NEO4J_PASSWORD = '$PW')"
    Write-Host ""
    Write-Step "Next step: Import dataset + rules to graph:"
    Write-Host "    `$env:NEO4J_PASSWORD = '$PW'  (if not set)"
    Write-Host "    python -m bate.scripts.import_to_neo4j"
}

# ==============================================================
#  check   : Only verify connectivity (for local/Desktop Neo4j)
# ==============================================================
"check" {
    Write-Step "Checking Neo4j connectivity (bolt://localhost:${BoltPort})..."
    $retry = 1
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $BoltPort)
        $c.Close()
        Write-OK "Bolt port ${BoltPort} is reachable. Neo4j Browser UI:  http://localhost:${HttpPort}"
        Write-Step "Run import: "
        Write-Host "    python -m bate.scripts.import_to_neo4j  --dry-run     # no write"
        Write-Host "    python -m bate.scripts.import_to_neo4j                 # actual write"
    } catch {
        Write-Fail "Bolt port ${BoltPort} NOT reachable."
        Write-Host "  Solutions:"
        Write-Host "    1) Docker path : .\scripts\start_neo4j.ps1 docker"
        Write-Host "    2) Local install: Open Neo4j Desktop -> Start the project -> wait ~20s -> rerun check"
    }
}

# ==============================================================
#  stop    : Stop container (only for docker path)
# ==============================================================
"stop" {
    $running = & docker ps -a --format '{{.Names}}' 2>$null | Where-Object { $_ -eq $ContainerName }
    if (-not $running) { Write-Warn "No container named '$ContainerName' found (local install stop via Neo4j Desktop UI)."; exit 0 }
    Write-Step "Stopping container '$ContainerName'..."
    & docker stop $ContainerName | Out-Null
    Write-OK "Container stopped."
}

# ==============================================================
#  logs    : Tail container logs
# ==============================================================
"logs" {
    $running = & docker ps -a --format '{{.Names}}' 2>$null | Where-Object { $_ -eq $ContainerName }
    if (-not $running) { Write-Fail "No container '$ContainerName'. Use 'start_neo4j.ps1 docker' first."; exit 0 }
    & docker logs --tail 50 --follow $ContainerName
}

} # end switch
