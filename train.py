################################################ Import necessary libraries ################################################

import sys
sys.path.append('/workspace/dongwoo/HGAIT/utils')
from utils import set_seed, calculate_regression_metrics, combined_loss

from Dataset_reg_std import FinancialGraphDatasetReg
from Dataloader import FinancialGraphDataLoader
from config import Config

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
from tqdm import tqdm  # For progress tracking
import wandb
import pickle
from collections import defaultdict
from tqdm import tqdm


from model import HGAIT

# setting CUDA_VISIBLE_DEVICES to use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import wandb
# Config 클래스 정의

# Create an instance of Config
config = Config()
if config.log_to_wandb:
    # Initialize wandb with the instance of Config-> 새로운 project 정의하였음!!
    wandb.init(project='', config=vars(config))
# Set the seed for reproducibility
set_seed(config.seed) 


# 데이터 경로 및 파일 목록 설정
file_names = sorted(os.listdir(config.data_dir))
date_list = [str(i[:10]) for i in file_names]

# Split the dataset
train_dates = date_list[:config.train_split]  # First 120 for training
val_dates = date_list[config.train_split:config.train_split + config.val_length]  # Next 24 for validation
test_dates = date_list[config.train_split + config.val_length:]  # Remaining for testing

# Create datasets
train_dataset = FinancialGraphDatasetReg(data_dir=config.data_dir, dates=train_dates)
val_dataset = FinancialGraphDatasetReg(data_dir=config.data_dir, dates=val_dates)
test_dataset = FinancialGraphDatasetReg(data_dir=config.data_dir, dates=test_dates)

# Create DataLoaders
train_loader = FinancialGraphDataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = FinancialGraphDataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = FinancialGraphDataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

device = torch.device('cuda' if config.use_cuda else 'cpu')

sequence_length = train_dataset[0].x.shape[2]
input_dim = train_dataset[0].x.shape[1]

# Model 선언
model = HGAIT(sequence_length=sequence_length, 
                  input_dim=input_dim, 
                  n_heads=config.n_heads, 
                  d_model=config.d_model, 
                  n_neighbors=config.n_neighbors,
                  n_layers = config.n_layers)
model.to(device)

# Loss Function
criterion = combined_loss
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4) # l2 regularization

# Learning rate scheduler 설정 (ReduceLROnPlateau)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, verbose=True)
early_stopping_counter = 0

# Best validation loss 기록
best_val_loss = float('inf')

# Training loop with early stopping and learning rate scheduler
for epoch in range(config.num_epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Apply tqdm for progress tracking
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Training]"):
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch.x)[0].squeeze()  # Output shape: (n_t, 1)
        
        # [t-L+1, t]까지에 대해 L개의 return에 대해 계산한 평균과 std를 활용해 denormalization
        denorm_preds = output * batch.std + batch.mean
        denorm_labels = batch.y * batch.std + batch.mean

        loss = criterion(denorm_preds, denorm_labels.float(), lambda_m = config.mse_lambda)  # Denormalized한 값을 토대로 Loss 계산
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        # Store denormalized predictions and labels for metrics -> 확인용
        all_preds.append(denorm_preds)
        all_labels.append(denorm_labels)
    torch.cuda.empty_cache()



    avg_loss = total_loss / len(train_loader)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    train_mse, train_mae = calculate_regression_metrics(all_preds, all_labels)
    print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {avg_loss:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}')

    # Validation loop
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    # Apply tqdm for progress tracking in validation
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Validation]"):
            batch = batch.to(device)
            output = model(batch.x).squeeze()  # Output shape: (B,)

            # Denormalize predictions and labels using node-wise mean/std values
            denorm_preds = output * batch.std + batch.mean
            denorm_labels = batch.y * batch.std + batch.mean

            loss = criterion(denorm_preds, denorm_labels.float(), lambda_m = config.mse_lambda)  # Denormalized MSELoss 계산
            val_loss += loss.item()

            # Store denormalized predictions and labels for metrics
            all_preds.append(denorm_preds)
            all_labels.append(denorm_labels)
    
    torch.cuda.empty_cache()

    avg_val_loss = val_loss / len(val_loader)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    val_mse, val_mae = calculate_regression_metrics(all_preds, all_labels)
    print(f'Validation Loss: {avg_val_loss:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}')

    # Log losses to wandb
    if config.log_to_wandb:
        wandb.log({
            'Train Split Length': config.train_split,
            'Validation Length': config.val_length,
            'learning_rate': config.learning_rate,
            'n_heads': config.n_heads,
            'd_model': config.d_model,
            'n_neighbors': config.n_neighbors,
            'seed': config.seed,
            'n_layers': config.n_layers,
            'Train Loss': avg_loss,
            'Train MSE': train_mse,
            'Train MAE': train_mae,
            'Validation Loss': avg_val_loss,
            'Validation MSE': val_mse,
            'Validation MAE': val_mae,
            'epoch': epoch + 1,
            'mse_lambda': config.mse_lambda
        })

    # Learning rate scheduling
    scheduler.step(avg_val_loss)

    # Early stopping 조건 확인
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0  # 카운터 초기화
        torch.save(model.state_dict(), config.best_model_path)
        print(f'Best model saved with validation loss: {best_val_loss:.4f}')
    else:
        early_stopping_counter += 1

    # Early stopping 조건 충족 시 학습 중단
    if early_stopping_counter >= config.early_stopping_patience:
        print(f'Early stopping triggered at epoch {epoch+1}')
        break

# Testing loop - Denormalization for both loss calculation and back-testing
model.load_state_dict(torch.load(config.best_model_path))  # Load the best model
model.eval()
test_loss = 0
all_preds = []
all_labels = []
final_results = {"Date": [], "stock_code": [], "pred": [], "real": []}

# Test 단계에서 사용할 denormalized된 예측값과 실제 레이블 저장
test_preds_by_date = defaultdict(list)  # 날짜별로 denormalized predictions 저장
test_labels_by_date = defaultdict(list)  # 날짜별로 denormalized labels 저장

# Apply tqdm for progress tracking in testing
with torch.no_grad():
    batch_num = 0
    for batch in tqdm(test_loader, desc="Testing"):
        batch = batch.to(device)
        output = model(batch.x).squeeze()  # Output shape: (B,)

        # Denormalize predictions and labels using node-wise mean/std values
        denorm_preds = output * batch.std + batch.mean
        denorm_labels = batch.y * batch.std + batch.mean

        loss = criterion(denorm_preds, denorm_labels.float(), lambda_m = config.mse_lambda)  # Denormalized MSELoss 계산    
        test_loss += loss.item()

        # Store denormalized predictions and labels for metrics
        all_preds.append(denorm_preds)
        all_labels.append(denorm_labels)

        # 배치 내의 각 샘플에 대해 날짜별로 예측 및 레이블 저장
        for i in range(denorm_preds.size(0)):  # 배치 내 모든 샘플을 반복
            test_preds_by_date[batch_num].append(denorm_preds[i].cpu().numpy())
            test_labels_by_date[batch_num].append(denorm_labels[i].cpu().numpy())
            final_results["Date"].append(batch_num)  # 날짜 저장
            final_results["stock_code"].append(batch.stock_keys[0][i])  # 종목 코드 저장
            final_results["pred"].append(denorm_preds[i].cpu().numpy())  # 예측값 저장
            final_results["real"].append(denorm_labels[i].cpu().numpy())  # 실제값 저장

        batch_num += 1
torch.cuda.empty_cache()

# Calculate final metrics
avg_test_loss = test_loss / len(test_loader)
all_preds = torch.cat(all_preds, dim=0)
all_labels = torch.cat(all_labels, dim=0)
test_mse, test_mae = calculate_regression_metrics(all_preds, all_labels)
print(f'Test Loss: {avg_test_loss:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}')

# Log test results to wandb
if config.log_to_wandb:
    wandb.log({
        'Test Loss': avg_test_loss,
        'Test MSE': test_mse,
        'Test MAE': test_mae
    })

# 결과를 config.result_dir에 저장
if not os.path.exists(config.result_dir):
    os.makedirs(config.result_dir)

# 저장 파일 경로 지정
save_path = os.path.join(config.result_dir, 
                         f"results_{config.train_split}_{config.num_epochs}_{config.learning_rate}_{config.n_heads}_{config.d_model}_{config.n_neighbors}_{config.seed}_{config.n_layers}_{config.mse_lambda}.pkl")

# 결과 저장 (denormalized predictions and labels by date)
with open(save_path, 'wb') as f:
    pickle.dump({
        'test_preds_by_date': test_preds_by_date,
        'test_labels_by_date': test_labels_by_date
    }, f)
    
import pandas as pd
df_results = pd.DataFrame(final_results)

# Ensure 'real' and 'pred' columns are float type
df_results['real'] = df_results['real'].astype(float)
df_results['pred'] = df_results['pred'].astype(float)

df_results.to_csv(f'results_{config.seed}_{config.learning_rate}_{config.n_heads}_{config.n_heads}_{config.n_layers}.csv', index=False)


print(f"Results saved to {save_path}")