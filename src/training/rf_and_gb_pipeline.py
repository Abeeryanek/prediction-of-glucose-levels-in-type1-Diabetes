import pandas as pd
import numpy as np
import glob
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import clarke_error_grid as ceg
from pathlib import Path
import warnings
import os
from grid import clarke_grid_import,create_bar_plot
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pickle

#check the path!
DATA_PATH ="data/bigideas"
# LOAD DATA
print("\n[1/12] LOADING DATA...")

file_list = glob.glob(os.path.join(DATA_PATH, "clean_patient_*.parquet"))
if not file_list:
    file_list = glob.glob("bent_*.parquet")
if not file_list:
    raise ValueError("No parquet files found!")

df = pd.concat([pd.read_parquet(f) for f in file_list], ignore_index=True)
df = df.sort_values(['Patient_ID', 'Timestamp']).reset_index(drop=True)
print(f"Loaded: {len(df)} rows from {df['Patient_ID'].nunique()} patients")
print(f"  Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

# 2. TEMPORAL FEATURES
df['Hour']            = df['Timestamp'].dt.hour
df['Minute']          = df['Timestamp'].dt.minute
df['DayOfWeek']       = df['Timestamp'].dt.dayofweek
df['MinFromMidnight'] = df['Hour'] * 60 + df['Minute']
df['Hour_sin']        = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos']        = np.cos(2 * np.pi * df['Hour'] / 24)

print("  Hour, Minute, DayOfWeek, MinFromMidnight, Hour_sin, Hour_cos")

# 3. INTERPOLATE SENSOR GAPS
print("\n[3/12] INTERPOLATING SENSOR GAPS...")

sensor_cols = [c for c in ['Heart_Rate', 'Acc_Vmu', 'EDA', 'Skin_Temp', 'BVP', 'IBI']
               if c in df.columns]
for col in sensor_cols:
    df[col] = (df.groupby('Patient_ID')[col]
                 .transform(lambda x: x.interpolate(method='linear', limit=6)))
print(f"  Interpolated {len(sensor_cols)} sensors: {sensor_cols}")

# 4. FOOD ROLLING WINDOWS
print("\n[4/12] LOADING FOOD FEATURES FROM PARQUET...")

food_features_all = [c for c in df.columns if any(
    c.startswith(nutrient) for nutrient in
    ['calorie_', 'total_carb_', 'dietary_fiber_', 'sugar_', 'protein_', 'total_fat_']
)]

food_features_carbs = [c for c in df.columns if any(
    c.startswith(nutrient) for nutrient in
    ['total_carb_']
)]
print(f"  Found {len(food_features_all)} total food features")
print(f"  Found {len(food_features_carbs)} carb-only features")

# 5. ACTIVITY ROLLING WINDOWS
print("\n[5/12] COMPUTING ACTIVITY ROLLING WINDOWS...")

ACT_WINDOWS       = {'30min': 6, '1h': 12, '2h': 24, '4h': 48}
activity_features = []

if 'Acc_Vmu' in df.columns:
    for label, periods in ACT_WINDOWS.items():
        col_mean = f'acc_vmu_mean_{label}'
        col_max  = f'acc_vmu_max_{label}'
        df[col_mean] = (df.groupby('Patient_ID')['Acc_Vmu']
                          .transform(lambda x: x.rolling(window=periods, min_periods=1).mean()))
        df[col_max]  = (df.groupby('Patient_ID')['Acc_Vmu']
                          .transform(lambda x: x.rolling(window=periods, min_periods=1).max()))
        activity_features += [col_mean, col_max]
    print(f"  Created {len(activity_features)} activity features from Acc_Vmu")

    # 6. PHYSIOLOGICAL VARIABILITY FEATURES
print("\n[6/12] COMPUTING PHYSIOLOGICAL VARIABILITY...")

physio_features_all = []
physio_features_heart_rate = []
physio_features_EDA = []
physio_features_Skin_Temp = []
physio_features_BVP = []
physio_features_IBI = []

if 'Heart_Rate' in df.columns:
    for label, periods in {'30min': 6, '1h': 12, '2h': 24}.items():
        col = f'hr_std_{label}'
        df[col] = (df.groupby('Patient_ID')['Heart_Rate']
                     .transform(lambda x: x.rolling(window=periods, min_periods=1).std()))
        physio_features_all.append(col)
        physio_features_heart_rate.append(col)
# Galvanic Skin Response
if 'EDA' in df.columns:
    for label, periods in {'30min': 6, '45min':9, '1h': 12}.items():
        col = f'eda_std_{label}'
        df[col] = (df.groupby('Patient_ID')['EDA']
                     .transform(lambda x: x.rolling(window=periods, min_periods=1).std()))
        physio_features_all.append(col)
        physio_features_EDA.append(col)
# Blood Volume Pulse
if 'BVP' in df.columns:
    for label, periods in {'30min': 6, '1h': 12}.items():
        col = f'bvp_std_{label}'
        df[col] = (df.groupby('Patient_ID')['BVP']
                     .transform(lambda x: x.rolling(window=periods, min_periods=1).std()))
        physio_features_all.append(col)
        physio_features_BVP.append(col)
if'IBI' in df.columns:
    for label, periods in {'30min': 6, '1h': 12}.items():
        col = f'ibi_std_{label}'
        df[col] = (df.groupby('Patient_ID')['IBI']
                     .transform(lambda x: x.rolling(window=periods, min_periods=1).std()))
        physio_features_all.append(col)
        physio_features_IBI.append(col)
if'Skin_Temp' in df.columns:
    for label, periods in {'30min': 6, '1h': 12}.items():
        col = f'skin_temp_std_{label}'
        df[col] = (df.groupby('Patient_ID')['Skin_Temp']
                     .transform(lambda x: x.rolling(window=periods, min_periods=1).std()))
        physio_features_all.append(col)
        physio_features_Skin_Temp.append(col)
print(f"  Created {len(physio_features_all)} variability features")
print(f"  Created {len(physio_features_heart_rate)} variability features")

# 7. GLUCOSE LAG FEATURES
print("\n[7/12] CREATING GLUCOSE LAG FEATURES...")

lag_features = []
for lag in [1, 2, 3, 6, 12]:
    col = f'glucose_lag_{lag}'
    df[col] = df.groupby('Patient_ID')['Glucose'].shift(lag)
    lag_features.append(col)

df['glucose_roc_1'] = df.groupby('Patient_ID')['Glucose'].diff(1)
df['glucose_roc_6'] = df.groupby('Patient_ID')['Glucose'].diff(6)
lag_features.extend(['glucose_roc_1', 'glucose_roc_6'])
print(f"  Created {len(lag_features)} glucose lag/change features")

# 8. BUILD TARGETS (15 / 30 / 45 MIN AHEAD)
HORIZONS = {'15min': 3, '30min': 6, '45min': 9}
for label, periods in HORIZONS.items():
    df[f'Target_{label}'] = df.groupby('Patient_ID')['Glucose'].shift(-periods)
    print(f"  {label}: {periods}  5min ahead")

# 9. PER-PATIENT 80/20 CHRONOLOGICAL SPLIT
print("\n[9/12] SPLITTING (80% TRAIN / 20% TEST PER PATIENT)...")

train_list, test_list = [], []
for pid, group in df.groupby('Patient_ID'):
    group     = group.sort_values('Timestamp').reset_index(drop=True)
    split_idx = int(len(group) * 0.8)
    train_list.append(group.iloc[:split_idx])
    test_list.append(group.iloc[split_idx:])

df_train = pd.concat(train_list, ignore_index=True)
df_test  = pd.concat(test_list,  ignore_index=True)
print(f"  Train: {len(df_train):,} rows ({len(df_train)/len(df)*100:.1f}%)")
print(f"  Test : {len(df_test):,} rows  ({len(df_test)/len(df)*100:.1f}%)")

# 10. FEATURE LIST + NORMALISATION
print("\n[10/12] PREPARING FEATURES...")

temporal_features = ['Hour','Minute','DayOfWeek','MinFromMidnight','Hour_sin','Hour_cos']
sensor_features   = sensor_cols
all_features      = (temporal_features + sensor_features + food_features_all + 
                     activity_features + physio_features_all+lag_features)
all_features      = [f for f in all_features if f in df_train.columns]
all_features_comparable=(temporal_features+ sensor_features  + food_features_carbs + 
                           activity_features + physio_features_heart_rate+ lag_features )
all_features_comparable = [f for f in all_features_comparable if f in df_train.columns]


print(f"\n  Feature summary:")
print(f"    Temporal      : {len(temporal_features)}")
print(f"    Sensors       : {len(sensor_features)}  → {sensor_features}")
print(f"    Food windows all  : {len(food_features_all)}")
print(f"    Food windows carbs  : {len(food_features_carbs)}")
print(f"    Activity      : {len(activity_features)}")
print(f"    Physiological all : {len(physio_features_all)}")
print(f"    Physiological heart rate : {len(physio_features_heart_rate)}")
print(f"    Glucose lags  : {len(lag_features)}")
print(f"    TOTAL         : {len(all_features)}")

dataset={
    'full':{},
    'comparable':{}}
for label in HORIZONS:
    target_col  = f'Target_{label}'
    train_clean = df_train.dropna(subset=[target_col] + all_features)
    test_clean  = df_test.dropna(subset=[target_col] + all_features)
    dataset['full'][label] = {
        'x_train': train_clean[all_features],
        'y_train': train_clean[target_col],
        'x_test':  test_clean[all_features],
        'y_test':  test_clean[target_col],
        'test_df': test_clean,
    }
    print(f"  {label}: train={len(train_clean):,}  test={len(test_clean):,}")

    target_col  = f'Target_{label}'
    train_clean_compare = df_train.dropna(subset=[target_col] + all_features_comparable)
    test_clean_compare  = df_test.dropna(subset=[target_col] + all_features_comparable)
    dataset['comparable'][label] = {
        'x_train': train_clean_compare[all_features_comparable],
        'y_train': train_clean_compare[target_col],
        'x_test':  test_clean_compare[all_features_comparable],
        'y_test':  test_clean_compare[target_col],
        'test_df': test_clean,
    }
    print(f"  {label}: train={len(train_clean_compare):,}  test={len(test_clean_compare):,}")
#-------------------------------------------data process done here-----------------------------------------------------------------------------------
rf  = RandomForestRegressor(n_estimators=300,  
                                       max_depth=20,
                                       min_samples_split=10,
                                       min_samples_leaf=4,
                                       max_features='sqrt',
                                       n_jobs=-1,
                                       random_state=42)

gb  = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=8,
                                   min_samples_split=10, min_samples_leaf=4,
                                   subsample=0.8, random_state=42)
horizen=['15min','30min','45min']
def calculate_clinical_weights(y_true):
    weights = np.ones(len(y_true))
    weights[y_true < 54]                          = 3.0
    weights[(y_true >= 54)  & (y_true < 70)]      = 2.5
    weights[(y_true > 180)  & (y_true <= 250)]    = 1.5
    weights[y_true > 250]                          = 2.0
    return weights
    
def run_pipeline(model,model_name,dataset):
    #scaling 

    results={}
    for subset in dataset:
        
        scaler = StandardScaler()
        scaler.fit(dataset[subset]['30min']['x_train'])
        for label in horizen:
            dataset[subset][label]['x_train_sc'] = scaler.transform(dataset[subset][label]['x_train'])
            dataset[subset][label]['x_test_sc']  = scaler.transform(dataset[subset][label]['x_test'])
        print("  StandardScaler fitted on 30min train, applied to all horizons")
        #train models with clinical weights
        print("\n[11/12] TRAINING MODELS WITH CLINICAL WEIGHTING...")
        print("="*80)
    
        for label in HORIZONS:
            print(f"\n{'='*60}")
            print(f"HORIZON: {label.upper()}")
            print(f"{'='*60}")

            x_tr = dataset[subset][label]['x_train_sc']
            y_tr = dataset[subset][label]['y_train']
            x_te = dataset[subset][label]['x_test_sc']
            y_te = dataset[subset][label]['y_test']

            print(f"  Sample weight distribution:")
            print(f"    Severe hypo  (<54)    : {(y_tr < 54).sum():4d} → weight 3.0")
            print(f"    Moderate hypo (54-70) : {((y_tr >= 54) & (y_tr < 70)).sum():4d} → weight 2.5")
            print(f"    Normal (70-180)       : {((y_tr >= 70) & (y_tr <= 180)).sum():4d} → weight 1.0")
            print(f"    Moderate hyper(180-250): {((y_tr > 180) & (y_tr <= 250)).sum():4d} → weight 1.5")
            print(f"    Severe hyper  (>250)  : {(y_tr > 250).sum():4d} → weight 2.0")
            train_weights = calculate_clinical_weights(y_tr.values)
            model.fit(x_tr, y_tr, sample_weight=train_weights)
            preds = model.predict(x_te)
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            mae  = mean_absolute_error(y_te, preds)
            r2   = r2_score(y_te, preds)
            mape = np.mean(np.abs((y_te - preds) / y_te)) * 100
            print(f"{model_name} → RMSE={rmse:.2f} MAE={mae:.2f} R²={r2:.4f} MAPE={mape:.2f}%")
            if label not in results:
                results[label] = {} 
            results[label][f'{model_name}_{subset}']= {
            'preds':   preds,
            'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape},
            'y_te':y_te
            }
    
    return results
    
feature_groups = {
    'physio_only':  lag_features + physio_features_all,
    'EDA'       :    lag_features+physio_features_EDA ,                          
    'heart_rate':    lag_features+physio_features_heart_rate ,                  
    'BVP'       :    lag_features+physio_features_BVP,                       
    'IBI'     :       lag_features+physio_features_IBI ,                   
    'food_only':    lag_features+food_features_all,
    'carbs_only':   lag_features+food_features_carbs,
    'activity':     lag_features+activity_features,
    'comparable':   all_features_comparable,
    'glucose':      ['Glucose']+lag_features,
    'all':          all_features
}


def run_ablation(model,feature_groups, df_train, df_test):
    ablation_results = {}
    for group_name, features in feature_groups.items():
        ablation_results[group_name] = {}
        print(f"\nABLATION: {group_name} ({len(features)} features)")

        for label in ['15min', '30min', '45min']:
            target_col  = f'Target_{label}'
            train_clean = df_train.dropna(subset=[target_col] + features)
            test_clean  = df_test.dropna(subset=[target_col] + features)

            scaler_abl  = StandardScaler()
            x_tr_abl    = scaler_abl.fit_transform(train_clean[features])
            x_te_abl    = scaler_abl.transform(test_clean[features])
            y_tr_abl    = train_clean[target_col]
            y_te_abl    = test_clean[target_col]
            w_abl       = calculate_clinical_weights(y_tr_abl.values)

            model.fit(x_tr_abl, y_tr_abl, sample_weight=w_abl)
            preds_abl = model.predict(x_te_abl)
            rmse_abl  = np.sqrt(mean_squared_error(y_te_abl, preds_abl))

            ablation_results[group_name][label] = rmse_abl
            print(f"  {label}: RMSE={rmse_abl:.2f}")
    return ablation_results 


#rf reults saved and clark grid plotted   
results = run_pipeline(rf, 'rf', dataset)
with open(f'results_rf.pkl', 'wb') as f:
    pickle.dump(results, f)
clarke_grid_import(results,'rf')

#ablation
results_ablation = run_ablation(rf,feature_groups,df_train,df_test)
with open(f'results_rf_ablation.pkl', 'wb') as f:
    pickle.dump(results, f)


#gb reults saved and clark grid plotted   
results = run_pipeline(gb, 'gb', dataset)
with open(f'results_gb.pkl', 'wb') as f:
    pickle.dump(results, f)
results_ablation = run_ablation(gb,feature_groups,df_train,df_test)
with open(f'results_gb_ablation.pkl', 'wb') as f:
    pickle.dump(results, f)
clarke_grid_import(results,'gb')

#bar plot for ablation or experimental features optional
create_bar_plot(results_ablation)

