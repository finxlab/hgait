import torch
from torch_geometric.data import Data, Dataset
import os
import pickle
from tqdm import tqdm

class FinancialGraphDatasetReg(Dataset):
    def __init__(self, data_dir, dates):
        """
        Dataset class to handle time-specific graph data with temporal and spatial node relationships.
        Args:
            data_dir (str): Path to the directory where preprocessed data is stored (per date).
            dates (list): List of dates to load the data from.
        """
        self.data_dir = data_dir
        self.dates = dates

    def __len__(self):
        """
        Returns the TOTAL number of available dates in the dataset.
        Returns:
            int: Number of dates in the dataset.
        """
        return len(self.dates)

    def __getitem__(self, idx):
        """
        Fetches the graph data for a specific date (idx).
        Args:
            idx (int)-> 몇번쨰 Date인지를 indicate하는 인덱스
        Returns:
            torch_geometric.data.Data-> Contains: features, labels, and adjacency matrix.
        """
        date = self.dates[idx]
        file_path = os.path.join(self.data_dir, f'{str(date)[:10]}.pkl')

        # Load preprocessed data for the specific date
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Node features (n_t nodes with time sequence data + technical indicators)
        features = data['features']  # Dict형태 -> {permno: feature sequence tensor}
        labels = data['labels']  # Dict 형태-> {permno: label sequence tensor (Up/Down)}
        mean_std = data['mean_std']  # Dict형태: {permno: {'mean': mean, 'std': std}}

        # 모든 n_t개의 stock(node)를 결집, 스택
        node_features = torch.stack([features[permno] for permno in sorted(features.keys())], dim=0)

        # Labels -> One-Step ahead에서의 label(returns)
        node_labels = torch.tensor([labels[permno][0][-1] for permno in sorted(labels.keys())], dtype=torch.float32)

        # Mean and standard deviation values for each node -> Instance Normalization을 위한 값들(원복이 필요)
        means = torch.tensor([mean_std[permno]['mean'] for permno in sorted(mean_std.keys())], dtype=torch.float32)
        stds = torch.tensor([mean_std[permno]['std'] for permno in sorted(mean_std.keys())], dtype=torch.float32)
        permnos = sorted(features.keys())

        # 아래와 같은 형식으로 torch.geometric.Data 형식으로의 Graph Data가 저장됨
        graph_data = Data(
            x=node_features,                 # Node features: (n_t * sequence length * 변수 개수)
            y=node_labels,                   # Node labels: Scaled returns for each node
            mean=means,                      # Mean values for each node(Stocks)
            std=stds,                        # Standard deviation values for each node(Stocks)
            stock_keys=permnos               # Stock keys
        )
        return graph_data
