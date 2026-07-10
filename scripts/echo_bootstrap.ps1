param(
    [ValidateSet("launch", "setup")]
    [string]$Mode = "launch",
    [string]$Model = "qwen2.5:3b",
    [switch]$IncludeTraining,
    [switch]$SkipModelPull
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
$TrainingRequirementsFile = Join-Path $ProjectRoot "requirements-training.txt"
$GgufRequirementsFile = Join-Path $ProjectRoot "requirements-gguf.txt"
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPythonw = Join-Path $VenvDir "Scripts\pythonw.exe"
$VenvVersionStamp = Join-Path $VenvDir ".python-version"
$RequirementsStamp = Join-Path $VenvDir ".requirements.sha256"
$TrainingRequirementsStamp = Join-Path $VenvDir ".requirements-training.sha256"
$GgufRequirementsStamp = Join-Path $VenvDir ".requirements-gguf.sha256"
$GgufModelPath = Join-Path $ProjectRoot "models\qwen-1_5b.gguf"
$SupportedPythonVersions = @("3.11", "3.12")

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Info([string]$Message) {
    Write-Host "[..] $Message" -ForegroundColor Gray
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-CommandExists([string]$Name) {
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Add-CommonToolPaths {
    $candidateDirs = @(
        "C:\Program Files\Ollama",
        "$env:LocalAppData\Programs\Python\Python311",
        "$env:LocalAppData\Programs\Python\Python311\Scripts",
        "$env:LocalAppData\Programs\Python\Python312",
        "$env:LocalAppData\Programs\Python\Python312\Scripts",
        "$env:ProgramFiles\Python311",
        "$env:ProgramFiles\Python311\Scripts",
        "$env:ProgramFiles\Python312",
        "$env:ProgramFiles\Python312\Scripts"
    )

    foreach ($dir in $candidateDirs) {
        if ((Test-Path $dir) -and ($env:Path -notlike "*$dir*")) {
            $env:Path = "$dir;$env:Path"
        }
    }
}

function Get-FileSha256([string]$Path) {
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($Path)
        try {
            $hashBytes = $sha256.ComputeHash($stream)
            return ([System.BitConverter]::ToString($hashBytes)).Replace("-", "")
        } finally {
            $stream.Dispose()
        }
    } finally {
        $sha256.Dispose()
    }
}

function Get-PythonVersion([string]$PythonExe) {
    try {
        $versionOutput = (& $PythonExe --version 2>&1 | Out-String).Trim()
        if ($LASTEXITCODE -eq 0 -and $versionOutput -match "Python\s+(\d+)\.(\d+)") {
            return "$($matches[1]).$($matches[2])"
        }
    } catch {
    }

    if ($PythonExe -match "Python311") {
        return "3.11"
    }
    if ($PythonExe -match "Python312") {
        return "3.12"
    }

    return $null
}

function Test-SupportedPythonVersion([string]$Version) {
    return $Version -and ($SupportedPythonVersions -contains $Version)
}

function Get-VenvRecordedPythonVersion {
    if (Test-Path $VenvVersionStamp) {
        return (Get-Content -Path $VenvVersionStamp -Raw).Trim()
    }

    $configPath = Join-Path $VenvDir "pyvenv.cfg"
    if (Test-Path $configPath) {
        $configContent = Get-Content -Path $configPath -Raw
        if ($configContent -match "version\s*=\s*(\d+)\.(\d+)") {
            return "$($matches[1]).$($matches[2])"
        }
    }

    return $null
}

function Resolve-AnyPython {
    Add-CommonToolPaths

    if (Test-CommandExists "py") {
        foreach ($selector in @("-3.11", "-3.12", "-3")) {
            try {
                $resolved = & py $selector -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $resolved) {
                    return ($resolved | Select-Object -First 1).Trim()
                }
            } catch {
            }
        }
    }

    foreach ($candidate in @("python.exe", "python")) {
        try {
            $resolved = & $candidate -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $resolved) {
                return ($resolved | Select-Object -First 1).Trim()
            }
        } catch {
        }
    }

    return $null
}

function Resolve-SystemPython {
    if (Test-CommandExists "py") {
        foreach ($selector in @("-3.11", "-3.12")) {
            try {
                $resolved = & py $selector -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $resolved) {
                    return ($resolved | Select-Object -First 1).Trim()
                }
            } catch {
            }
        }
    }

    foreach ($candidatePath in @(
        "$env:LocalAppData\Programs\Python\Python311\python.exe",
        "$env:LocalAppData\Programs\Python\Python312\python.exe",
        "$env:ProgramFiles\Python311\python.exe",
        "$env:ProgramFiles\Python312\python.exe"
    )) {
        if (Test-Path $candidatePath) {
            return $candidatePath
        }
    }

    $genericPython = Resolve-AnyPython
    if (-not $genericPython) {
        return $null
    }

    $genericVersion = Get-PythonVersion $genericPython
    if (Test-SupportedPythonVersion $genericVersion) {
        return $genericPython
    }

    return $null
}

function Resolve-OllamaExe {
    Add-CommonToolPaths

    $cmd = Get-Command "ollama" -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    foreach ($path in @(
        "C:\Program Files\Ollama\ollama.exe",
        "$env:LocalAppData\Programs\Ollama\ollama.exe"
    )) {
        if (Test-Path $path) {
            return $path
        }
    }

    return $null
}

function Resolve-OllamaCommand([string]$FallbackExe) {
    if (Test-CommandExists "ollama") {
        return "ollama"
    }
    return $FallbackExe
}

function Test-OllamaProcessRunning {
    return $null -ne (Get-Process ollama -ErrorAction SilentlyContinue)
}

function Resolve-AppPythonCommand([string]$FallbackPython) {
    if (Test-CommandExists "python") {
        return "python"
    }
    return $FallbackPython
}

function Resolve-AppPythonwCommand([string]$FallbackPython) {
    if (Test-CommandExists "pythonw") {
        return "pythonw"
    }
    return $FallbackPython
}

function Invoke-SelfElevatedIfNeeded([bool]$NeedAdmin) {
    if (-not $NeedAdmin -or (Test-IsAdministrator)) {
        return
    }

    Write-Warn "Для установки недостающих компонентов потребуется запуск от имени администратора."
    $argumentList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`"",
        "-Mode", $Mode,
        "-Model", $Model
    )
    if ($IncludeTraining) {
        $argumentList += "-IncludeTraining"
    }
    if ($SkipModelPull) {
        $argumentList += "-SkipModelPull"
    }

    $process = Start-Process -FilePath "powershell.exe" -ArgumentList $argumentList -Verb RunAs -Wait -PassThru
    exit $process.ExitCode
}

function Install-Python {
    Write-Step "Установка Python"

    if (Test-CommandExists "winget") {
        foreach ($packageId in @("Python.Python.3.11", "Python.Python.3.12")) {
            Write-Info "Пробую установить $packageId через winget"
            & winget install --id $packageId -e --silent --accept-package-agreements --accept-source-agreements
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "Python установлен через winget: $packageId"
                return
            }
        }
    }

    if (Test-CommandExists "choco") {
        foreach ($packageId in @("python311", "python312", "python")) {
            Write-Info "Пробую установить $packageId через choco"
            & choco install $packageId --yes
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "Python установлен через choco: $packageId"
                return
            }
        }
    }

    throw "Не удалось установить Python автоматически. Нужен winget или choco."
}

function Install-Ollama {
    Write-Step "Установка Ollama"

    if (Test-CommandExists "winget") {
        Write-Info "Пробую установить Ollama через winget"
        & winget install --id Ollama.Ollama -e --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Ollama установлена через winget"
            return
        }
    }

    if (Test-CommandExists "choco") {
        Write-Info "Пробую установить Ollama через choco"
        & choco install ollama --yes
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Ollama установлена через choco"
            return
        }
    }

    throw "Не удалось установить Ollama автоматически. Нужен winget или choco."
}

function Ensure-ProjectFolders {
    Write-Step "Проверка структуры проекта"

    foreach ($relativePath in @(
        "logs",
        "knowledge_input",
        "data\processed",
        "data\teacher",
        "models",
        "models\echo-lora"
    )) {
        $fullPath = Join-Path $ProjectRoot $relativePath
        if (-not (Test-Path $fullPath)) {
            New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
            Write-Info "Создана папка: $relativePath"
        }
    }

    Write-Ok "Структура проекта готова"
}

function Ensure-Venv([string]$SystemPython) {
    Write-Step "Проверка виртуального окружения"

    $systemVersion = Get-PythonVersion $SystemPython
    if (-not (Test-SupportedPythonVersion $systemVersion)) {
        throw "Неподдерживаемая версия Python: $systemVersion. Нужен Python 3.11 или 3.12."
    }

    if (Test-Path $VenvPython) {
        $venvVersion = Get-VenvRecordedPythonVersion
        if ($venvVersion -and $venvVersion -ne $systemVersion) {
            Write-Warn "Найдена .venv на Python $venvVersion. Пересоздаю окружение под Python $systemVersion."
            Remove-Item -LiteralPath $VenvDir -Recurse -Force
        } elseif (-not $venvVersion) {
            Write-Warn "Не удалось определить версию существующей .venv, оставляю её без пересоздания."
        }
    }

    if (-not (Test-Path $VenvPython)) {
        Write-Info "Создаю .venv"
        & $SystemPython -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) {
            throw "Не удалось создать .venv"
        }
        Set-Content -Path $VenvVersionStamp -Value $systemVersion -Encoding ascii
    }

    Write-Ok "Виртуальное окружение готово"
}

function Invoke-PipInstallWithFallback([string[]]$PipArguments, [string]$StepName) {
    $attempts = @(
        @{
            Name = "${StepName}: стандартный запуск"
            Args = $PipArguments
        },
        @{
            Name = "${StepName}: повтор с trusted-host"
            Args = $PipArguments + @(
                "--trusted-host", "pypi.org",
                "--trusted-host", "files.pythonhosted.org",
                "--trusted-host", "pypi.python.org",
                "--prefer-binary"
            )
        },
        @{
            Name = "${StepName}: повтор через официальный индекс PyPI"
            Args = $PipArguments + @(
                "--index-url", "https://pypi.org/simple",
                "--trusted-host", "pypi.org",
                "--trusted-host", "files.pythonhosted.org",
                "--trusted-host", "pypi.python.org",
                "--prefer-binary"
            )
        }
    )

    foreach ($attempt in $attempts) {
        Write-Info $attempt.Name
        & $VenvPython -m pip @($attempt.Args)
        if ($LASTEXITCODE -eq 0) {
            return
        }
    }

    throw "$StepName завершился ошибкой после всех попыток."
}

function Ensure-Requirements {
    Write-Step "Проверка Python-зависимостей"

    if (-not (Test-Path $RequirementsFile)) {
        throw "Не найден requirements.txt"
    }

    $requirementsHash = Get-FileSha256 $RequirementsFile
    $installedHash = ""
    if (Test-Path $RequirementsStamp) {
        $installedHash = (Get-Content -Path $RequirementsStamp -Raw).Trim()
    }

    if ($installedHash -ne $requirementsHash) {
        Invoke-PipInstallWithFallback -PipArguments @("install", "--upgrade", "pip", "setuptools", "wheel") -StepName "Обновление pip"
        Invoke-PipInstallWithFallback -PipArguments @("install", "-r", $RequirementsFile) -StepName "Установка зависимостей Python"

        Set-Content -Path $RequirementsStamp -Value $requirementsHash -Encoding ascii
        Write-Ok "Python-зависимости установлены"
    } else {
        Write-Ok "Python-зависимости уже актуальны"
    }
}

function Ensure-TrainingRequirements {
    if (-not $IncludeTraining) {
        Write-Info "Расширенный training stack пропущен. Для него используйте setup_training_stack.bat."
        return
    }

    if (-not (Test-Path $TrainingRequirementsFile)) {
        Write-Warn "Не найден requirements-training.txt, расширенный training stack пропущен."
        return
    }

    Write-Step "Проверка training-зависимостей"

    $requirementsHash = Get-FileSha256 $TrainingRequirementsFile
    $installedHash = ""
    if (Test-Path $TrainingRequirementsStamp) {
        $installedHash = (Get-Content -Path $TrainingRequirementsStamp -Raw).Trim()
    }

    if ($installedHash -eq $requirementsHash) {
        Write-Ok "Training-зависимости уже актуальны"
        return
    }

    Invoke-PipInstallWithFallback -PipArguments @("install", "-r", $TrainingRequirementsFile) -StepName "Установка training-зависимостей"
    Set-Content -Path $TrainingRequirementsStamp -Value $requirementsHash -Encoding ascii
    Write-Ok "Training-зависимости установлены"
}

function Ensure-OptionalGgufSupport {
    if (-not $IncludeTraining) {
        Write-Info "GGUF-поддержка пропущена в базовом режиме setup."
        return
    }

    if (-not (Test-Path $GgufModelPath)) {
        Write-Info "GGUF-модель не найдена, optional-пакет llama-cpp-python пока не нужен."
        return
    }

    if (-not (Test-Path $GgufRequirementsFile)) {
        Write-Warn "Есть GGUF-модель, но отсутствует requirements-gguf.txt."
        return
    }

    Write-Step "Проверка optional-поддержки GGUF"

    $requirementsHash = Get-FileSha256 $GgufRequirementsFile
    $installedHash = ""
    if (Test-Path $GgufRequirementsStamp) {
        $installedHash = (Get-Content -Path $GgufRequirementsStamp -Raw).Trim()
    }

    if ($installedHash -eq $requirementsHash) {
        Write-Ok "GGUF-зависимости уже актуальны"
        return
    }

    try {
        Invoke-PipInstallWithFallback -PipArguments @("install", "-r", $GgufRequirementsFile) -StepName "Установка GGUF-зависимостей"
        Set-Content -Path $GgufRequirementsStamp -Value $requirementsHash -Encoding ascii
        Write-Ok "GGUF-зависимости установлены"
    } catch {
        Write-Warn "GGUF-поддержка не установлена автоматически: $($_.Exception.Message)"
        Write-Warn "Основной режим с Ollama продолжит работать. Для GGUF можно повторить setup позже."
    }
}

function Test-PythonImports([string]$PythonCommand) {
    Write-Step "Проверка импортов"

    Push-Location $ProjectRoot
    try {
        $previousPythonPath = $env:PYTHONPATH
        $env:PYTHONPATH = $ProjectRoot
        $probeFile = Join-Path $env:TEMP "echo_bootstrap_probe.py"
        $probeCode = @(
            "import tkinter, sqlite3",
            "print('imports-ok')",
            "import echo_app.gui",
            "print('gui-import-ok')"
        ) -join "`r`n"
        [System.IO.File]::WriteAllText($probeFile, $probeCode + "`r`n", [System.Text.Encoding]::UTF8)

        & $PythonCommand $probeFile
        if ($LASTEXITCODE -ne 0) {
            throw "Проверка импортов завершилась с ошибкой"
        }
    } finally {
        $env:PYTHONPATH = $previousPythonPath
        Pop-Location
    }

    Write-Ok "Импорты и GUI проверены"
}

function Ensure-OllamaResponsive([string]$OllamaCommand) {
    Write-Step "Проверка Ollama"

    $responsive = $false
    try {
        & $OllamaCommand list *> $null
        $responsive = ($LASTEXITCODE -eq 0)
    } catch {
        $responsive = $false
    }

    if (-not $responsive -and (Test-OllamaProcessRunning)) {
        Write-Warn "Команда Ollama недоступна из текущего PATH, но процесс уже запущен. Продолжаю запуск."
        return
    }

    if (-not $responsive) {
        Write-Warn "Ollama не отвечает, пробую запустить сервис"
        Start-Process -FilePath $OllamaCommand -ArgumentList "serve" -WindowStyle Hidden | Out-Null
        for ($attempt = 0; $attempt -lt 15; $attempt++) {
            Start-Sleep -Seconds 2
            try {
                & $OllamaCommand list *> $null
                if ($LASTEXITCODE -eq 0) {
                    $responsive = $true
                    break
                }
            } catch {
            }
        }
    }

    if (-not $responsive -and (Test-OllamaProcessRunning)) {
        Write-Warn "Команда Ollama всё ещё недоступна, но процесс активен. Проверку API пропускаю."
        return
    }

    if (-not $responsive) {
        throw "Ollama установлена, но сервис не отвечает."
    }

    Write-Ok "Ollama доступна"
}

function Ensure-OllamaModel([string]$OllamaCommand, [string]$ModelName) {
    Write-Step "Проверка модели Ollama"

    try {
        $listOutput = (& $OllamaCommand list 2>$null) | Out-String
    } catch {
        if (Test-OllamaProcessRunning) {
            Write-Warn "Не удалось проверить список моделей через команду Ollama. Пропускаю эту проверку, потому что процесс Ollama уже активен."
            return
        }
        throw
    }

    if ($listOutput -match "(?m)^\s*$([regex]::Escape($ModelName))\s") {
        Write-Ok "Модель уже загружена: $ModelName"
        return
    }

    if ($SkipModelPull) {
        Write-Warn "Модель $ModelName не найдена, но загрузка пропущена флагом -SkipModelPull"
        return
    }

    Write-Info "Скачиваю модель $ModelName через Ollama"
    & $OllamaCommand pull $ModelName
    if ($LASTEXITCODE -ne 0) {
        throw "Не удалось загрузить модель $ModelName"
    }

    Write-Ok "Модель загружена: $ModelName"
}

function Launch-App([string]$ModelName, [string]$PythonCommand) {
    Write-Step "Запуск Эхо"

    $env:ECHO_OLLAMA_MODEL = $ModelName
    $env:PYTHONPATH = $ProjectRoot

    Start-Process -FilePath $PythonCommand -WorkingDirectory $ProjectRoot -ArgumentList "Alter_Echo.py" | Out-Null
    Write-Ok "Эхо запущено"
}

try {
    Write-Host "ЭХО: bootstrap mode = $Mode" -ForegroundColor White
    Ensure-ProjectFolders

    $anyPython = Resolve-AnyPython
    $systemPython = Resolve-SystemPython
    $ollamaExe = Resolve-OllamaExe
    Invoke-SelfElevatedIfNeeded (-not $systemPython -or -not $ollamaExe)

    if (-not $systemPython -and $anyPython) {
        $detectedVersion = Get-PythonVersion $anyPython
        if ($detectedVersion) {
            Write-Warn "Найден Python $detectedVersion ($anyPython), но для Эхо нужен Python 3.11 или 3.12."
        }
    }

    if (-not $systemPython) {
        Install-Python
        $systemPython = Resolve-SystemPython
    }
    if (-not $systemPython) {
        throw "Python не найден даже после установки."
    }
    Write-Ok "Найден Python: $systemPython"
    Write-Info "Версия Python: $(Get-PythonVersion $systemPython)"

    if (-not $ollamaExe) {
        Install-Ollama
        $ollamaExe = Resolve-OllamaExe
    }
    if (-not $ollamaExe) {
        throw "Ollama не найдена даже после установки."
    }
    Write-Ok "Найдена Ollama: $ollamaExe"

    $appPython = Resolve-AppPythonCommand $systemPython
    $appPythonw = Resolve-AppPythonwCommand $appPython
    $ollamaCommand = Resolve-OllamaCommand $ollamaExe

    Ensure-Venv -SystemPython $systemPython
    Ensure-Requirements
    Ensure-TrainingRequirements
    Ensure-OptionalGgufSupport
    Test-PythonImports -PythonCommand $appPython
    Ensure-OllamaResponsive -OllamaCommand $ollamaCommand
    Ensure-OllamaModel -OllamaCommand $ollamaCommand -ModelName $Model

    Write-Step "Итоговая проверка"
    Write-Ok "Проект готов к работе"
    Write-Info "GUI: Alter_Echo.py"
    Write-Info "Учитель Ollama: $Model"
    if ($IncludeTraining) {
        Write-Info "Расширенный training stack включён."
        Write-Info "Примечание: при первом обучении или сборке Hugging Face и llama.cpp могут скачать дополнительные файлы автоматически."
    } else {
        Write-Info "Базовый режим готов: GUI + Ollama + qwen2.5:3b."
    }

    if ($Mode -eq "launch") {
        Launch-App -ModelName $Model -PythonCommand $appPythonw
    }

    exit 0
} catch {
    Write-Host ""
    Write-Host "BOOTSTRAP ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
