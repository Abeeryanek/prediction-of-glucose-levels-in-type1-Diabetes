import os
import sys
import time
import pandas as pd
import numpy as np
from google.colab import files

print("Installing multi-threaded download engine (aria2)...")
os.system("apt-get install -y aria2 > /dev/null")

# Constants
CHUNKSIZE   = 100_000
FOOD_COLS   = ['calorie', 'total_carb', 'dietary_fiber', 'sugar', 'protein', 'total_fat']
WINDOWS     = {'2h': 24, '8h': 96, '24h': 288}
BASE_URL    = "https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3"

# Starting at Patient 8
for i in range(1, 17):
    pid = f"{i:03d}"
    output_file = f"clean_patient_{pid}.parquet"

    print(f"\n{'='*50}")
    print(f"PROCESSING PATIENT {pid} (WITH BVP - MULTI-THREADED)")
    print(f"{'='*50}")

    # ==========================================
    # 0. SMART RESUME / PARQUET CHECK
    # ==========================================
    if os.path.exists(output_file):
        print(f"  ✓ {output_file} already exists!")
        print(f"  Skipping data processing and jumping straight to download...")
        try:
            files.download(output_file)
            time.sleep(5)
        except Exception as e:
            print(f"Browser blocked the automatic download: {e}")
        os.system("rm -rf /content/*.csv")
        continue

    # =====================================================================
    # INDIVIDUAL PROCESSING (aria2 parallel downloading)
    # =====================================================================

    # 1. Dexcom
    file_name = f"Dexcom_{pid}.csv"
    if not os.path.exists(file_name):
        print(f"  ↓ Downloading {file_name}...")
        os.system(f'aria2c -x 8 -s 8 -q -U "Mozilla/5.0" {BASE_URL}/{pid}/{file_name}')

    try:
        df_dex = pd.read_csv(file_name)
        df_dex = df_dex[df_dex['Event Type'] == 'EGV'].copy()
        df_dex['Timestamp'] = pd.to_datetime(df_dex['Timestamp (YYYY-MM-DDThh:mm:ss)'], errors='coerce')
        df_dex['Glucose']   = pd.to_numeric(df_dex['Glucose Value (mg/dL)'], errors='coerce')
        df_dex = (df_dex[['Timestamp', 'Glucose']]
                    .dropna().set_index('Timestamp').sort_index())
        df_dex = df_dex.resample('5min').mean().dropna()

        anchor = df_dex.index
        df_dex['Patient_ID']      = pid
        df_dex['Hour']            = anchor.hour
        df_dex['Minute']          = anchor.minute
        df_dex['DayOfWeek']       = anchor.weekday
        df_dex['MinFromMidnight'] = anchor.hour * 60 + anchor.minute
        print(f"  ✓ Dexcom: {len(df_dex)} rows")
    except Exception as e:
        print(f"  ✗ Dexcom failed — {e}. Skipping Patient {pid}.")
        os.system("rm -rf /content/*.csv")
        continue

    # 2. Heart Rate
    file_name = f"HR_{pid}.csv"
    if not os.path.exists(file_name):
        print(f"  ↓ Downloading {file_name}...")
        os.system(f'aria2c -x 8 -s 8 -q -U "Mozilla/5.0" {BASE_URL}/{pid}/{file_name}')

    try:
        hr_chunks = []
        for chunk in pd.read_csv(file_name, chunksize=CHUNKSIZE):
            chunk['datetime'] = pd.to_datetime(chunk.iloc[:, 0], format='mixed', errors='coerce')
            hr_chunks.append(chunk.set_index('datetime').sort_index()
                                    .iloc[:, 0:1]
                                    .apply(pd.to_numeric, errors='coerce')
                                    .resample('5min').mean())
        df_dex['Heart_Rate'] = (pd.concat(hr_chunks).resample('5min').mean()
                                    .iloc[:, 0].reindex(anchor).values)
        print("  ✓ Heart Rate")
    except Exception as e:
        df_dex['Heart_Rate'] = np.nan
        print("  ✗ Heart Rate failed")

    # 3. ACC (Movement / Exercise Vector)
    file_name = f"ACC_{pid}.csv"
    if not os.path.exists(file_name):
        print(f"  ↓ Downloading {file_name}...")
        os.system(f'aria2c -x 8 -s 8 -q -U "Mozilla/5.0" {BASE_URL}/{pid}/{file_name}')

    try:
        acc_chunks = []
        for chunk in pd.read_csv(file_name, chunksize=CHUNKSIZE):
            chunk['datetime'] = pd.to_datetime(chunk.iloc[:, 0], format='mixed', errors='coerce')
            chunk = chunk.set_index('datetime').sort_index()
            x = pd.to_numeric(chunk.iloc[:, 0], errors='coerce')
            y = pd.to_numeric(chunk.iloc[:, 1], errors='coerce')
            z = pd.to_numeric(chunk.iloc[:, 2], errors='coerce')
            vmu = pd.DataFrame({'acc_vmu': np.sqrt(x**2 + y**2 + z**2)})
            acc_chunks.append(vmu.resample('5min').mean())
        df_dex['Acc_Vmu'] = (pd.concat(acc_chunks).resample('5min').mean()
                                .iloc[:, 0].reindex(anchor).values)
        print("  ✓ Accelerometer (ACC)")
    except Exception as e:
        df_dex['Acc_Vmu'] = np.nan
        print("  ✗ ACC failed")

    # 4. EDA
    file_name = f"EDA_{pid}.csv"
    if not os.path.exists(file_name):
        print(f"  ↓ Downloading {file_name}...")
        os.system(f'aria2c -x 8 -s 8 -q -U "Mozilla/5.0" {BASE_URL}/{pid}/{file_name}')

    try:
        eda_chunks = []
        for chunk in pd.read_csv(file_name, chunksize=CHUNKSIZE):
            chunk['datetime'] = pd.to_datetime(chunk.iloc[:, 0], format='mixed', errors='coerce')
            eda_chunks.append(chunk.set_index('datetime').sort_index()
                                    .iloc[:, 0:1]
                                    .apply(pd.to_numeric, errors='coerce')
                                    .resample('5min').mean())
        df_dex['EDA'] = (pd.concat(eda_chunks).resample('5min').mean()
                            .iloc[:, 0].reindex(anchor).values)
        print("  ✓ EDA")
    except Exception as e:
        df_dex['EDA'] = np.nan
        print("  ✗ EDA failed")

    # 5. Skin Temp
    file_name = f"TEMP_{pid}.csv"
    if not os.path.exists(file_name):
        print(f"  ↓ Downloading {file_name}...")
        os.system(f'aria2c -x 8 -s 8 -q -U "Mozilla/5.0" {BASE_URL}/{pid}/{file_name}')

    try:
        temp_chunks = []
        for chunk in pd.read_csv(file_name, chunksize=CHUNKSIZE):
            chunk['datetime'] = pd.to_datetime(chunk.iloc[:, 0], format='mixed', errors='coerce')
            temp_chunks.append(chunk.set_index('datetime').sort_index()
                                     .iloc[:, 0:1]
                                     .apply(pd.to_numeric, errors='coerce')
                                     .resample('5min').mean())
        df_dex['Skin_Temp'] = (pd.concat(temp_chunks).resample('5min').mean()
                                    .iloc[:, 0].reindex(anchor).values)
        print("  ✓ Skin Temp")
    except Exception as e:
        df_dex['Skin_Temp'] = np.nan
        print("  ✗ Skin Temp failed")

    # 6. BVP - THE MASSIVE FILE (Now using 8 connections)
    file_name = f"BVP_{pid}.csv"
    if not os.path.exists(file_name):
        print(f"  ↓ Downloading {file_name} (Using 8 parallel connections to bypass throttling)...")
        os.system(f'aria2c -x 8 -s 8 -q -U "Mozilla/5.0" {BASE_URL}/{pid}/{file_name}')

    try:
        bvp_chunks = []
        for chunk in pd.read_csv(file_name, chunksize=CHUNKSIZE):
            chunk['datetime'] = pd.to_datetime(chunk.iloc[:, 0], format='mixed', errors='coerce')
            bvp_chunks.append(chunk.set_index('datetime').sort_index()
                                    .iloc[:, 0:1]
                                    .apply(pd.to_numeric, errors='coerce')
                                    .resample('5min').mean())
        df_dex['BVP'] = (pd.concat(bvp_chunks).resample('5min').mean()
                            .iloc[:, 0].reindex(anchor).values)
        print("  ✓ BVP successfully fused!")
    except Exception as e:
        df_dex['BVP'] = np.nan
        print(f"  ✗ BVP failed — {e}")

    # 7. IBI
    file_name = f"IBI_{pid}.csv"
    if not os.path.exists(file_name):
        print(f"  ↓ Downloading {file_name}...")
        os.system(f'aria2c -x 8 -s 8 -q -U "Mozilla/5.0" {BASE_URL}/{pid}/{file_name}')

    try:
        ibi_chunks = []
        for chunk in pd.read_csv(file_name, chunksize=CHUNKSIZE):
            chunk['datetime'] = pd.to_datetime(chunk.iloc[:, 0], format='mixed', errors='coerce')
            ibi_chunks.append(chunk.set_index('datetime').sort_index()
                                    .iloc[:, 0:1]
                                    .apply(pd.to_numeric, errors='coerce')
                                    .resample('5min').mean())
        df_dex['IBI'] = (pd.concat(ibi_chunks).resample('5min').mean()
                            .iloc[:, 0].reindex(anchor).values)
        print("  ✓ IBI")
    except Exception as e:
        df_dex['IBI'] = np.nan
        print("  ✗ IBI failed")

    # 8. Food Log
    file_name = f"Food_Log_{pid}.csv"
    if not os.path.exists(file_name):
        print(f"  ↓ Downloading {file_name}...")
        os.system(f'aria2c -x 8 -s 8 -q -U "Mozilla/5.0" {BASE_URL}/{pid}/{file_name}')

    food_df = pd.DataFrame(0.0, index=anchor, columns=FOOD_COLS)
    try:
        df_food = pd.read_csv(file_name)
        df_food.columns       = df_food.columns.str.strip().str.lower().str.replace(' ', '_')
        df_food['time_begin'] = pd.to_datetime(df_food['time_begin'], format='mixed', errors='coerce')
        df_food = df_food.dropna(subset=['time_begin']).set_index('time_begin').sort_index()
        for col in FOOD_COLS:
            if col in df_food.columns:
                df_food[col] = pd.to_numeric(df_food[col], errors='coerce').fillna(0)
                resampled    = df_food[[col]].resample('5min').sum()[col]
                food_df[col] = resampled.reindex(anchor, fill_value=0).values
        print("  ✓ Food Log")
    except Exception as e:
        print("  ✗ Food Log failed")

    for col in FOOD_COLS:
        for label, periods in WINDOWS.items():
            df_dex[f'{col}_{label}'] = (food_df[col]
                                          .ewm(halflife=periods, min_periods=1)
                                          .mean().values)

    # 9. Save parquet locally inside Colab
    patient_df = df_dex.reset_index()
    patient_df.to_parquet(output_file, index=False)
    print(f"Processed: {len(patient_df)} rows, {len(patient_df.columns)} columns")

    # Now safe to wipe raw CSVs to save RAM/Disk space for the NEXT patient
    os.system("rm -rf /content/*.csv")
    print("Raw files deleted from Colab server to free space")

    # =====================================================================
    # THE BULLETPROOF DOWNLOAD SECTION
    # =====================================================================
    print(f"\n PATIENT {pid} PROCESSED SUCCESSFULLY!")
    print("Triggering download to your physical computer...")

    try:
        files.download(output_file)
        print(f"SUCCESS! Check your computer's 'Downloads' folder for {output_file}")
        # 5 second pause so the browser doesn't block the download before the next loop starts
        time.sleep(5)
    except Exception as e:
        print(f"Browser blocked the automatic download: {e}")
        print(f"Please click the Folder icon on the left to manually download {output_file}.")

print("\n" + "="*50)
print("BATCH FINISHED!")
print("="*50)