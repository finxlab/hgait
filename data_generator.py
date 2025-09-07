import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import rank_returns_diff

#########################################

# Load the dataset
df = pd.read_csv('/workspace/dongwoo/HGAIT/raw_data/CRSP.csv', index_col=0)
df.rename(columns=lambda x: x.replace('_', ' '), inplace=True)

df.rename(columns={'Closing Price': 'Last Trade Price'}, inplace=True)
df['Lowest Bid Price '] = df['Lowest Bid Price'].abs()
df['Highest Ask Price'] = df['Highest Ask Price'].abs()
df['Last Trade Price'] = df['Last Trade Price'].abs()
df['Trading Volume'] = df['Trading Volume'].abs() 

# Moving Average (3, 12 period)
df['MA_3'] = df.groupby('Permanent Company ID Unique')['Last Trade Price'].transform(lambda x: x.rolling(window=3).mean().where(x.notna()))
df['MA_12'] = df.groupby('Permanent Company ID Unique')['Last Trade Price'].transform(lambda x: x.rolling(window=12).mean().where(x.notna()))

# Momentum (12 period)
df['Momentum_12'] = df.groupby('Permanent Company ID Unique')['Last Trade Price'].transform(lambda x: (x - x.shift(12)).where(x.notna()))

# Stochastic Oscillator (9 period)
df['Lowest_Min'] = df.groupby('Permanent Company ID Unique')['Lowest Bid Price'].transform(lambda x: x.rolling(window=9).min())
df['Highest_Max'] = df.groupby('Permanent Company ID Unique')['Highest Ask Price'].transform(lambda x: x.rolling(window=9).max())
df['Stochastic_Oscillator'] = ((df['Last Trade Price'] - df['Lowest_Min']) / (df['Highest_Max'] - df['Lowest_Min']) * 100).where(df['Last Trade Price'].notna())

# MACD (12 and 26 period)
df['EMA_12'] = df.groupby('Permanent Company ID Unique')['Last Trade Price'].transform(lambda x: x.ewm(span=12, adjust=False).mean().where(x.notna()))
df['EMA_26'] = df.groupby('Permanent Company ID Unique')['Last Trade Price'].transform(lambda x: x.ewm(span=26, adjust=False).mean().where(x.notna()))
df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df.groupby('Permanent Company ID Unique')['MACD_Line'].transform(lambda x: x.ewm(span=9, adjust=False).mean().where(x.notna()))

# RSI (9 period)
df['Price_Change'] = df.groupby('Permanent Company ID Unique')['Last Trade Price'].transform(lambda x: x.diff().where(x.notna()))
df['Gain'] = df['Price_Change'].where(df['Price_Change'] > 0, 0)
df['Loss'] = -df['Price_Change'].where(df['Price_Change'] < 0, 0)
df['Avg_Gain'] = df.groupby('Permanent Company ID Unique')['Gain'].transform(lambda x: x.rolling(window=9).mean().where(x.notna()))
df['Avg_Loss'] = df.groupby('Permanent Company ID Unique')['Loss'].transform(lambda x: x.rolling(window=9).mean().where(x.notna()))
df['RS'] = df['Avg_Gain'] / df['Avg_Loss']
df['RSI'] = 100 - (100 / (1 + df['RS']))

result_df = df[['date', 'Permanent Company ID Unique','Lowest Bid Price','Highest Ask Price','Last Trade Price','Trading Volume','Return', # t 시점에서의 return은 그 전 시점 대비 변화량을 의미. 미래참조가 아님!!
                'MA_3', 'MA_12','Momentum_12', 'Stochastic_Oscillator', 'MACD_Line', 'Signal_Line', 'RSI']]

# 명시적으로 .loc 사용하여 열을 수정
result_df.loc[:, 'Return'] = result_df['Return'].fillna(0)

# 'Next Return'을 'Return' 열에서 한 칸 밀어서 생성
result_df.loc[:, 'Next Return'] = result_df.groupby('Permanent Company ID Unique')['Return'].shift(-1)

# 'Next Return'의 NaN 값을 앞의 값으로 채움 (ffill) -> 이거 떄문에 그럼. 따라서 마지막은 제외해 주는 것이 맞겠다!
result_df.loc[:, 'Next Return'] = result_df.groupby('Permanent Company ID Unique')['Next Return'].transform(lambda x: x.ffill())


result_df = result_df.groupby('date').apply(rank_returns_diff)
#drop the rows with 'Trading Volume' == 0 -> 아주 중요!!!!
result_df = result_df[result_df['Trading Volume'] != 0]


result_df = result_df.drop(columns = ['Rank'])
result_df = result_df.reset_index(level=0, drop=True).reset_index()
result_df = result_df.set_index(keys='date')

result_df.index = pd.to_datetime(result_df.index)

corr_cal_dates = result_df.index[(result_df.index<=pd.to_datetime('2023-12-31'))&(result_df.index>=pd.to_datetime('1995-01-01'))].unique()
total_date_list = result_df.index.unique()
timeframe = 24


from tqdm import tqdm
import numpy as np

results = []

# 필요한 기간 내의 데이터 미리 필터링
for date in tqdm(corr_cal_dates, desc='Generating Training Data'):
    corr_seq_end = date  # in datetime format
    corr_seq_end_index = np.where(total_date_list == corr_seq_end)[0][0]
    corr_seq_start_index = corr_seq_end_index - timeframe + 1
    corr_seq_start = total_date_list[corr_seq_start_index]
    
    # 기간 내의 데이터 미리 필터링
    date_range_data = result_df[(result_df.index >= corr_seq_start) & (result_df.index <= corr_seq_end)]
    
    # 해당 날짜의 고유 component 가져오기
    components = result_df[result_df.index == date]['Permanent Company ID Unique'].unique()
    
    component_dict = {}
        
    for component in components:
        component_data = date_range_data[date_range_data['Permanent Company ID Unique'] == component]
        
        # 필터된 데이터의 길이 및 NaN 검사
        if len(component_data) == timeframe and not component_data.isna().any().any():
            component_dict[component] = component_data

    # component_dict이 비어 있지 않으면 results에 추가
    if component_dict:
        results.append(component_dict)


def check_for_nans_in_results(results):
    nan_found = False
    
    for i, component_dict in enumerate(results):
        for component, data in component_dict.items():
            if data.isna().sum().sum() > 0:  # 더 빠른 NaN 체크
                print(f"NaN found in component {component} at index {i}")
                nan_found = True
                break  # NaN 발견 시, 내부 루프 바로 종료
        if nan_found:
            break  # NaN 발견 시, 외부 루프도 바로 종료

    if not nan_found:
        print("No NaNs found in the results")

check_for_nans_in_results(results)

total_date_list = total_date_list.tolist()
mask_data_list = []

# 각 day마다 데이터를 처리
for day in tqdm(range(len(results)), desc='Processing Results for Regression'):
    corr_to_use_date = corr_cal_dates[day]
    corr_to_use_date_idx = total_date_list.index(corr_to_use_date)
    corr_cal_start_idx = corr_to_use_date_idx - timeframe + 1

    dts = total_date_list[corr_cal_start_idx:corr_to_use_date_idx + 1]
    all_components_df = pd.concat([results[day][key] for key in results[day].keys()], ignore_index=True)
    all_components_df.drop(columns=['index'], inplace=True)

    features_to_scale = [feature for feature in all_components_df.columns if feature not in ['date', 'Permanent Company ID Unique', 'Ranked_Return', 'Next Return']]
    unique_permnos = all_components_df['Permanent Company ID Unique'].unique()

    scaled_data_list = []
    feature_sequences = {}
    label_sequences = {}
    mask_data_list = []
    next_return_real = []
    day_stats = {}  # Mean and std for each permno

    for permno in unique_permnos:
        group = all_components_df[all_components_df['Permanent Company ID Unique'] == permno].copy()

        permno_scaler = None
        for feature in features_to_scale:
            scaler = StandardScaler()
            group[feature] = scaler.fit_transform(group[[feature]])

            if feature == 'Return':
                permno_scaler = scaler
                day_stats[permno] = {
                    'mean': scaler.mean_[0],
                    'std': np.sqrt(scaler.var_[0])
                }

                # Standardize Next Return using mean and std
                group['Next Return'] = (group['Next Return'].values - day_stats[permno]['mean']) / day_stats[permno]['std']
        
        feature_sequence = group[features_to_scale].values.tolist()
        next_return_scaled = group['Next Return'].values.tolist()

        last_row_idx = group.index[-1]
        mask_data = group.at[last_row_idx, 'Ranked_Return']
        mask_data_list.append(mask_data)
        next_return_real.append(group.at[last_row_idx, 'Next Return'])

        if permno not in feature_sequences:
            feature_sequences[permno] = []
            label_sequences[permno] = []

        feature_sequences[permno].extend(feature_sequence)
        label_sequences[permno].append(next_return_scaled)

        scaled_data_list.append(group)

    final_scaled_df = pd.concat(scaled_data_list, ignore_index=True)
    feature_sequences = {key: torch.tensor(val, dtype=torch.float32).transpose(0, 1) for key, val in feature_sequences.items()}
    label_sequences = {key: torch.tensor(val, dtype=torch.float32) for key, val in label_sequences.items()}
    
    next_return_data = final_scaled_df.groupby('Permanent Company ID Unique').tail(1)['Next Return'].values

    try:
        result_dicts = {
            'features': feature_sequences,
            'real_returns': torch.tensor(next_return_real, dtype=torch.float32),
            'labels': label_sequences,
            'mask': torch.tensor(mask_data_list, dtype=torch.float32),
            'mean_std': day_stats,
        }
        
        output_dir = '/workspace/dongwoo/HGAIT/data'
        with open(os.path.join(output_dir, f'{str(corr_to_use_date)[:10]}.pkl'), 'wb') as f:
            pickle.dump(result_dicts, f)

    except Exception as e:
        raise ValueError(f"Error in creating result dictionary: {str(e)}")
