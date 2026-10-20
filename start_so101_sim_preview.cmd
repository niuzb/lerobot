@echo off
setlocal
cd /d "%~dp0"
for /f %%I in ('powershell.exe -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "SIM_PREVIEW_RUN=%%I"
"C:\Users\Admin\.local\bin\uv.exe" run --offline --no-project --python "C:\Users\Admin\miniconda3\envs\lerobot312\python.exe" --no-managed-python --with mujoco==3.8.1 --with scipy==1.18.1 python -X utf8 so101_sim_preview.py --output "outputs\sim_experiment_%SIM_PREVIEW_RUN%"
set "SIM_PREVIEW_RESULT=%ERRORLEVEL%"
echo.
echo Output: %~dp0outputs\sim_experiment_%SIM_PREVIEW_RUN%
echo Preview: simulation_preview_h264.mp4
exit /b %SIM_PREVIEW_RESULT%
