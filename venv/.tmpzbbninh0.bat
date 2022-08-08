@ECHO OFF
@SET PYTHONIOENCODING=utf-8
@SET PYTHONUTF8=1
@FOR /F "tokens=2 delims=:." %%A in ('chcp') do for %%B in (%%A) do set "_CONDA_OLD_CHCP=%%B"
@chcp 65001 > NUL
@CALL "C:\ProgramData\Anaconda3\condabin\conda.bat" activate "C:\Users\2022713\.conda\envs\LG_AImers"
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@python c:\Users\2022713\.vscode\extensions\ms-python.python-2022.4.0\pythonFiles\lib\python\debugpy\adapter
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@chcp %_CONDA_OLD_CHCP%>NUL
