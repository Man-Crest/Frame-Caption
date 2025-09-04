@echo off
setlocal ENABLEDELAYEDEXPANSION

rem Detect GPU by checking nvidia-smi
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  nvidia-smi -L >nul 2>&1
  if %ERRORLEVEL% EQU 0 (
    set COMPOSE_FILE=docker-compose.yml
    echo GPU detected. Using %COMPOSE_FILE%.
  ) else (
    set COMPOSE_FILE=docker-compose.cpu.yml
    echo No GPU detected. Using %COMPOSE_FILE%.
  )
) else (
  set COMPOSE_FILE=docker-compose.cpu.yml
  echo No GPU detected. Using %COMPOSE_FILE%.
)

if "%~1" NEQ "" (
  rem Pass-through custom docker compose commands, e.g., run_auto.bat down
  docker compose -f "%COMPOSE_FILE%" %*
) else (
  rem Default: build and start detached
  docker compose -f "%COMPOSE_FILE%" build
  if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
  docker compose -f "%COMPOSE_FILE%" up -d
)

endlocal

