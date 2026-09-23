@echo off
REM ============================================================
REM  Anemia Kuku — Run Manual: Retrain Hb (Step 1 + 2)
REM  Jalankan dari folder anemia-app (double-click atau via cmd)
REM ============================================================
cd /d "%~dp0..\.."          REM -> anemia-app

echo ================================================
echo  STEP 1: Retrain 3 varian (open/closed/mean)
echo ================================================
python experiments\hb_newdata\p2_train.py
if errorlevel 1 goto :err

echo.
echo ================================================
echo  STEP 2: Validasi lintas-domain MSU-250
echo ================================================
python experiments\hb_newdata\p3_eval_msu.py --model experiments\hb_newdata\models\open\elasticnet_model.joblib   --tag open
python experiments\hb_newdata\p3_eval_msu.py --model experiments\hb_newdata\models\closed\elasticnet_model.joblib --tag closed
python experiments\hb_newdata\p3_eval_msu.py --model experiments\hb_newdata\models\mean\elasticnet_model.joblib   --tag mean

echo.
echo ================================================
echo  SELESAI. Cek ringkasan di atas, lalu:
echo    python experiments\hb_newdata\p4_integrate.py --model-dir experiments\hb_newdata\models\mean --apply
echo ================================================
pause
exit /b 0

:err
echo.
echo [ERROR] Ada yang gagal. Periksa pesan di atas.
pause
exit /b 1