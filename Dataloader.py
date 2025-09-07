from torch_geometric.loader import DataLoader


####### Graph DataLoader for Financial Graphs #######

class FinancialGraphDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=True):
        """
        DataLoader class for loading the FinancialGraphDataset.

        Args:
            dataset (torch_geometric.data.Dataset): 로드할 그래프 데이터셋.
            batch_size (int): Number of graphs per batch. -> 1이 default. 한번에 처리될 데이터셋은 각 date t에 존재하는 n_t개의 노드들임.
            shuffle (bool): Whether to shuffle the data at the start of each epoch -> 날짜 순서대로가 아닌, 랜덤한 순서로 데이터를 처리하고 싶을 때 True로 설정.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Define the PyG DataLoader for graph data
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

    def __iter__(self):
        """
        Make the class iterable to return batches of data.
        """
        return iter(self.loader)

    def __len__(self):
        """
        Return the number of batches per epoch.
        """
        return len(self.loader)
