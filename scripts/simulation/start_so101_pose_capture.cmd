@echo off
cd /d "%~dp0..\.." || exit /b 1
"C:\Users\Admin\.local\bin\uv.exe" run --no-project --python "C:\Users\Admin\miniconda3\envs\lerobot312\python.exe" --no-managed-python python -X utf8 "%~dp0so101_pose_capture.py" --square-mm 17 --port 8766
pause
