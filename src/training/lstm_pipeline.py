import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import clarke_error_grid as ceg
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader #??
from grid import clarke_grid_import
warnings.filterwarnings('ignore')

DATA_PATH = str("bental patients")

# LOAD DATA
print("\n[1/12] LOADING DATA...")
model='lstm'

file_list = glob.glob(os.path.join(DATA_PATH, "clean_patient_*.parquet"))
if not file_list:
    file_list = glob.glob("bent_*.parquet")
if not file_list:
    raise ValueError("No parquet files found!")

df = pd.concat([pd.read_parquet(f) for f in file_list], ignore_index=True)
df = df.sort_values(['Patient_ID', 'Timestamp']).reset_index(drop=True)
print(f"Loaded: {len(df):,} rows from {df['Patient_ID'].nunique()} patients")
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

# 10. LSTM 3D SEQUENCES & NORMALIZATION
print("\n[10/12] PREPARING LSTM 3D SEQUENCES...")
temporal_features = ['Hour','Minute','DayOfWeek','MinFromMidnight','Hour_sin','Hour_cos']
sensor_features   = ['Glucose'] + sensor_cols
all_features      = (temporal_features + sensor_features + food_features_all +
                     activity_features + physio_features_all + lag_features)
all_features      = [f for f in all_features if f in df_train.columns]
all_features      = (temporal_features + sensor_features + food_features_all + 
                     activity_features + physio_features_all+lag_features)
all_features      = [f for f in all_features if f in df_train.columns]
all_features_comparable=(temporal_features+ sensor_features  + food_features_carbs + 
                           activity_features + physio_features_heart_rate + lag_features)
all_features_comparable = [f for f in all_features_comparable if f in df_train.columns]
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
    'glucose':   lag_features,
    'all':          all_features
}
def create_3d_sequences(X_scaled, y_series, seq_length=12):
    Xs, ys = [], []
    for i in range(seq_length - 1, len(X_scaled)):
        Xs.append(X_scaled[i - seq_length + 1 : i + 1])
        ys.append(y_series.iloc[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

SEQ_LENGTH = 12


datasets={}


    
for label in HORIZONS:
    target_col  = f'Target_{label}'
    train_clean = df_train.dropna(subset=[target_col] + all_features).reset_index(drop=True)
    test_clean  = df_test.dropna(subset=[target_col] + all_features).reset_index(drop=True)

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(train_clean[all_features])
    X_test_2d  = scaler.transform(test_clean[all_features])

    X_train_3d, y_train = create_3d_sequences(X_train_2d, train_clean[target_col], SEQ_LENGTH)
    X_test_3d, y_test   = create_3d_sequences(X_test_2d, test_clean[target_col], SEQ_LENGTH)

    datasets[label] = {
        f'X_train': X_train_3d,
        f'y_train': y_train,
        f'X_test':  X_test_3d,
        f'y_test':  y_test,
        f'test_df': test_clean.iloc[SEQ_LENGTH - 1:].reset_index(drop=True),
        f'scaler':  scaler
    }
datasets_comparable= {}
for label in HORIZONS:
    target_col  = f'Target_{label}'
    train_clean = df_train.dropna(subset=[target_col] + all_features_comparable).reset_index(drop=True)
    test_clean  = df_test.dropna(subset=[target_col] + all_features_comparable).reset_index(drop=True)

    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(train_clean[all_features_comparable])
    X_test_2d  = scaler.transform(test_clean[all_features_comparable])

    X_train_3d, y_train = create_3d_sequences(X_train_2d, train_clean[target_col], SEQ_LENGTH)
    X_test_3d, y_test   = create_3d_sequences(X_test_2d, test_clean[target_col], SEQ_LENGTH)

    datasets_comparable[label] = {
        f'X_train': X_train_3d,
        f'y_train': y_train,
        f'X_test':  X_test_3d,
        f'y_test':  y_test,
        f'test_df': test_clean.iloc[SEQ_LENGTH - 1:].reset_index(drop=True),
        f'scaler':  scaler
    }   



# 11. PYTORCH LSTM ARCHITECTURE + CLINICAL WEIGHTING

print("\n[11/12] TRAINING WEIGHTED LSTM MODELS...")
print("="*80)




class GlucoseDataset(Dataset):
    def __init__(self, x, y, weights):
        self.x = x
        self.y = y
        self.w = weights

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]
class GlucoseLSTM(nn.Module):
    def __init__(self,input_dim, hidden_dim=64,num_layers=2, dropout=0.3):       
        super().__init__()         
        self.lstm = nn.LSTM(input_dim, hidden_dim,num_layers, batch_first=True,dropout=dropout)   
        self.fc = nn.Linear(hidden_dim, 1)   

    def forward(self, x):          
        lstm_out, _ = self.lstm(x) 
        return self.fc(lstm_out[:, -1, :])  
def calculate_clinical_weights(y_true):
        weights = np.ones(len(y_true), dtype=np.float32)
        weights[y_true < 54] = 3.0
        weights[(y_true >= 54) & (y_true < 70)] = 2.5
        weights[(y_true > 180) & (y_true <= 250)] = 1.5
        weights[y_true > 250] = 2.0
        return weights
def train_weighted_lstm(x_train, y_train, sample_weights, input_dim, epochs=100, patience=15):
        #evalaution 85% was ist oe what 
        split = int(len(x_train) * 0.85)
        xt = torch.tensor(x_train[:split])
        yt = torch.tensor(y_train[:split]).unsqueeze(1)
        wt = torch.tensor(sample_weights[:split]).unsqueeze(1)

        xv = torch.tensor(x_train[split:])
        yv = torch.tensor(y_train[split:]).unsqueeze(1)
        wv = torch.tensor(sample_weights[split:]).unsqueeze(1)

        model = GlucoseLSTM(input_dim=input_dim)
        criterion = nn.MSELoss(reduction='none') # None so we can apply custom weights
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        #check this 
        train_loader = DataLoader(GlucoseDataset(xt, yt, wt), batch_size=64, shuffle=True)

        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            for xb, yb, wb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                
                loss = (criterion(preds, yb) * wb).mean()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(xv)
                val_loss = (criterion(val_preds, yv) * wv).mean().item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(best_weights)
        return model

def run_pipeline(model_name,datasets,datasets_comparable):
    results = {}
    for label in ['15min', '30min', '45min']:
        results[label] = {}
        print(f"\n{'='*60}")
        print(f"HORIZON: {label.upper()}")
        print(f"{'='*60}")

        x_tr = datasets[label]['X_train']
        y_tr = datasets[label]['y_train']
        x_te = datasets[label]['X_test']
        y_te = datasets[label]['y_test']
        train_weights = calculate_clinical_weights(y_tr)
        print(f"  Training Weighted LSTM Neural Network...")
        lstm_model = train_weighted_lstm(x_tr, y_tr, train_weights, input_dim=len(all_features))
        lstm_model.eval()
        with torch.no_grad():
            lstm_preds = lstm_model(torch.tensor(x_te)).numpy().flatten()

        lstm_rmse = np.sqrt(mean_squared_error(y_te, lstm_preds))
        lstm_mae  = mean_absolute_error(y_te, lstm_preds)
        lstm_r2   = r2_score(y_te, lstm_preds)
        lstm_mape = np.mean(np.abs((y_te - lstm_preds) / y_te)) * 100

        print(f"    LSTM → RMSE={lstm_rmse:.2f}  MAE={lstm_mae:.2f}  R²={lstm_r2:.4f}  MAPE={lstm_mape:.2f}%")

        results[label][f'{model_name}_full'] = {
            'preds':   lstm_preds,
            'metrics': {'rmse': lstm_rmse, 'mae': lstm_mae, 
            'r2': lstm_r2, 'mape': lstm_mape},
            'y_te':    y_te
        }
        x_tr_comparable = datasets_comparable[label]['X_train']
        y_tr_comparable = datasets_comparable[label]['y_train']
        x_te_comparable = datasets_comparable[label]['X_test']
        y_te_comparable = datasets_comparable[label]['y_test']
        train_weights_comparable = calculate_clinical_weights(y_tr_comparable)

        print(f"  Training Weighted Comparable LSTM Neural Network...")
        lstm_model_comparable = train_weighted_lstm(x_tr_comparable, y_tr_comparable, train_weights_comparable, input_dim=len(all_features_comparable))
        with torch.no_grad():
            lstm_preds_comparable=lstm_model_comparable(torch.tensor(x_te_comparable)).numpy().flatten()
            lstm_rmse_comparable = np.sqrt(mean_squared_error(y_te_comparable, lstm_preds_comparable))
            lstm_mae_comparable  = mean_absolute_error(y_te_comparable, lstm_preds_comparable)
            lstm_r2_comparable  = r2_score(y_te_comparable, lstm_preds_comparable)
            lstm_mape_comparable = np.mean(np.abs((y_te_comparable - lstm_preds_comparable) / y_te_comparable)) * 100

        print(f"    LSTM Comparable → RMSE={lstm_rmse_comparable:.2f}  MAE={lstm_mae_comparable:.2f}  R²={lstm_r2_comparable:.4f}  MAPE={lstm_mape_comparable:.2f}%")
        results[label][f'{model_name}_comparable'] = {
            'preds':   lstm_preds_comparable,
            'metrics': {'rmse': lstm_rmse_comparable, 
            'mae': lstm_mae_comparable,
            'r2': lstm_r2_comparable, 
            'mape': lstm_mape_comparable},
            'y_te':    y_te_comparable
        }
    return results

def run_ablation():
    results_ablation = {} 
    for group_name, features in feature_groups.items():
        results_ablation[group_name] = {}
        for label in HORIZONS:
            target_col  = f'Target_{label}'
            train_clean = df_train.dropna(subset=[target_col] + features).reset_index(drop=True)
            test_clean  = df_test.dropna(subset=[target_col] + features).reset_index(drop=True)

            scaler = StandardScaler()
            X_train_2d = scaler.fit_transform(train_clean[features])
            X_test_2d  = scaler.transform(test_clean[features])

            X_train_3d, y_train = create_3d_sequences(X_train_2d, train_clean[target_col], SEQ_LENGTH)
            X_test_3d, y_test   = create_3d_sequences(X_test_2d, test_clean[target_col], SEQ_LENGTH)
            x_tr = X_train_3d
            y_tr = y_train
            x_te = X_test_3d
            y_te = y_test
            train_weights = calculate_clinical_weights(y_tr)
            print(f"  Training Weighted LSTM Neural Network...")
            lstm_model = train_weighted_lstm(x_tr, y_tr, train_weights, input_dim=len(features))
            lstm_model.eval()
            with torch.no_grad():
                lstm_preds = lstm_model(torch.tensor(x_te)).numpy().flatten()

            lstm_rmse = np.sqrt(mean_squared_error(y_te, lstm_preds))

            print(f"    LSTM → RMSE={lstm_rmse:.2f}")

            results_ablation[group_name][label]= lstm_rmse
        
          
    return results_ablation
     
#lstm    
results = run_pipeline('lstm',datasets,datasets_comparable)
with open(f'results_lstm.pkl', 'wb') as f:
    pickle.dump(results, f)
results_ablation = run_ablation()
with open(f'results_lstm_ablation.pkl', 'wb') as f:
    pickle.dump(results_ablation, f)

clarke_grid_import(results,'lstm')



   