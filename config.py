import torch

#### 실험 시 조절할 파라미터 관리하는 변수들 ####
# 향후 yaml 등으로 관리

class Config:
    data_dir = '/workspace/dongwoo/HGAIT/data'
    train_split = 180 # 1995~2009. 총 5년.
    val_length = 24 # 2010 ~2011. # validation: 2년, test: 2012~2023: 12년. 
    batch_size = 1 # 큰 하나의 그래프를 load하는 형식
    num_epochs = 200
    learning_rate = 5e-4
    n_heads = 8     
    d_model = 128
    n_neighbors = 0.05 # 소수점 기준으로 표현. 0.1-> 10%. 노드 수가 많으므로 
    use_cuda = torch.cuda.is_available() # GPU 사용 여부
    seed = 7 
    mse_lambda = 0.5 ## 통일!!!!-> Validation Loss 상에서 하나의 term에 잠식되지 않도록 하기 위함.
    #loss_type = nn.MSELoss()
    n_layers = 3
    best_model_path = f'/workspace/dongwoo/HGAIT/model/best_model_{train_split}_{num_epochs}_{learning_rate}_{n_heads}_{d_model}_{n_neighbors}_{seed}_{n_layers}_{mse_lambda}.pth'
    result_dir = f'/workspace/dongwoo/HGAIT/results'
    # Early stopping을 위한 변수 설정
    early_stopping_patience = 10  
    log_to_wandb = True # option to log
