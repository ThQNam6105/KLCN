@echo off
chcp 65001 >nul
title Hệ Thống Phát Hiện Xâm Nhập Mạng Đa Mô Hình (NIDS)
echo ========================================================
echo   Hệ Thống Phát Hiện Xâm Nhập Mạng Đa Mô Hình (NIDS)
echo   Khóa luận tốt nghiệp cử nhân ATTT-KLCN54
echo ========================================================
echo.
echo Đang khởi động ứng dụng web Streamlit...
echo.

REM Kiểm tra môi trường ảo .venv
if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    streamlit run app.py
)

pause
