import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
from torch import Tensor
from math import sqrt
import math
from typing import Literal
from torch_geometric.nn import GATConv

    
class TimeMixing(nn.Module):
    def __init__(self, 
                 num_features: int, 
                 sequence_length: int, 
                 d_model: int, 
                 dropout_rate: float = 0.4, 
                 mode: Literal['lstm', 'mlp', 'gru'] = 'gru'):
        """
        TimeMixing Layer supporting flexible modes: 'lstm', 'mlp', 'gru'.

        Args:
            num_features: Number of input features (variables).
            sequence_length: Length of the time series sequence.
            d_model: Desired embedding dimension for each feature.
            dropout_rate: Dropout rate.
            mode: Embedding mode ('lstm', 'mlp', or 'gru'). -> 실험용
        """
        super(TimeMixing, self).__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.mode = mode
        #self.batch_norm = TimeBatchNorm2d((sequence_length, num_features))


        # Define layers based on the mode
        if mode == 'mlp':
            self.time_mixers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(sequence_length, d_model),  # Direct projection
                    nn.GELU(),
                    nn.Dropout(dropout_rate)
                )
                for _ in range(num_features)
            ])
        elif mode == 'lstm':
            self.time_mixers = nn.ModuleList([
                nn.LSTM(input_size=sequence_length, hidden_size=d_model, batch_first=True)
                for _ in range(num_features)
            ])
        elif mode == 'gru':
            self.time_mixers = nn.ModuleList([
                nn.GRU(input_size=sequence_length, hidden_size=d_model, batch_first=True)
                for _ in range(num_features)
            ])
        else:
            raise ValueError("Invalid mode. Choose from ['lstm', 'mlp', 'gru'].")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [num_nodes, num_features, seq_len].
        Returns:
            Tensor of shape [num_nodes, num_features, d_model].
        """
        num_nodes, num_features, seq_len = x.shape
        assert num_features == self.num_features, "Input features do not match model configuration."

        feature_embeddings = []
        for i in range(num_features):
            feature_data = x[:, i, :]  # Shape: [num_nodes, seq_len]

            if self.mode == 'mlp':
                # MLP-based projection
                feature_embedding = self.time_mixers[i](feature_data.permute(0,2,1))
            elif self.mode in ['lstm', 'gru']:
                # LSTM/GRU expects input shape [batch_size, seq_len, feature_dim]
                feature_data = feature_data.unsqueeze(-1)  # Shape: [num_nodes, seq_len, 1]
                rnn_out, _ = self.time_mixers[i](feature_data.permute(0,2,1))  # Shape: [num_nodes, seq_len, d_model]
                feature_embedding = rnn_out.squeeze()  # Use the last hidden state

            feature_embeddings.append(feature_embedding)

        # Stack feature embeddings: [num_nodes, num_features, d_model]
        stacked_embeddings = torch.stack(feature_embeddings, dim=1)
        
    
        return stacked_embeddings

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.4):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, C_Q, H, D_Q = queries.shape
        scale = self.scale or 1. / sqrt(D_Q)
        scores = torch.einsum("bche,bshe->bhcs", queries, keys)
        A = torch.softmax(scale * scores, dim=-1)
        A = self.dropout(A)
        V = torch.einsum("bhcs,bshd->bchd", A, values)
        return V.contiguous(), A

class FeatureImportanceCalculation(nn.Module):
    def __init__(self, model_dim: int, dropout_rate: float = 0.4):
        super(FeatureImportanceCalculation, self).__init__()
        self.fc1 = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(model_dim, 1)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        w = self.act(self.fc1(x))  
        w = self.dropout(w)        
        w = self.fc2(w)            
        w = F.softmax(w, dim=1)    
        return w

class FeatureImportanceTokenizedBlock(nn.Module): # 인풋으로 최종적으로 얻음 feature importance Calculation값을 인풋으로 받아서 fusion한다.
    def __init__(self):
        super(FeatureImportanceTokenizedBlock, self).__init__()

    def forward(self, x: Tensor, feature_importance: Tensor) -> Tensor:
        x_weighted = feature_importance * x
        x_combined = torch.sum(x_weighted, dim=1)
        return x_combined 

class InvertedAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(InvertedAttentionLayer, self).__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.layer_norm_q = nn.LayerNorm(d_keys * n_heads)  
        self.layer_norm_k = nn.LayerNorm(d_keys * n_heads)  
        self.layer_norm_v = nn.LayerNorm(d_values * n_heads)  
        self.n_heads = n_heads  

    def forward(self, queries, keys, values, attn_mask=None):
        B, C, att_dim = queries.shape
        H = self.n_heads
        queries = self.layer_norm_q(self.query_projection(queries)).view(B, C, H, -1)
        keys = self.layer_norm_k(self.key_projection(keys)).view(B, C, H, -1)
        values = self.layer_norm_v(self.value_projection(values)).view(B, C, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, C, -1)
        return self.out_projection(out), attn

class VariableAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.4):
        super(VariableAttentionLayer, self).__init__()
        self.attention = InvertedAttentionLayer(FullAttention(), d_model, n_heads)
        self.feature_importance_calc = FeatureImportanceCalculation(d_model)
        self.layer_norm = nn.LayerNorm(d_model)  
        self.dropout_after_attn = nn.Dropout(dropout_rate)
        self.activation_fn = nn.GELU()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_original = x.clone()
        x, attn_scores = self.attention(x, x, x)
        x = self.activation_fn(x)
        x = self.dropout_after_attn(x)
        x = self.layer_norm(x + x_original) # x shape: (node개수, 변수개수, d_model)
        #x = self.layer_norm(x) # 변화한 부분
        #x = x + x_original # 변화한 부분
        feature_importance = self.feature_importance_calc(x)
        return x, feature_importance


class GatedAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.4):
        super(GatedAttentionLayer, self).__init__()
        
        # GAT layers for self, top, and bottom attention
        self.self_gat = GATConv(
            in_channels=d_model,
            out_channels=d_model // n_heads,
            heads=n_heads,
            dropout=dropout_rate,
            concat=True
        )
        
        self.top_gat = GATConv(
            in_channels=d_model,
            out_channels=d_model // n_heads,
            heads=n_heads,
            dropout=dropout_rate,
            concat=True
        )
        
        self.bottom_gat = GATConv(
            in_channels=d_model,
            out_channels=d_model // n_heads,
            heads=n_heads,
            dropout=dropout_rate,
            concat=True
        )
        
        # Importance calculation layers
        self.importance_calc = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(3)  # self, top, bottom
        ])
        
        # Output processing
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def _create_edge_index(self, indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Create edge_index tensor for GAT from indices."""
        batch_indices = torch.arange(num_nodes, device=indices.device)
        source = batch_indices.repeat_interleave(indices.size(1))
        target = indices.reshape(-1)
        edge_index = torch.stack([source, target])
        return edge_index

    def forward(self, x: torch.Tensor, top_n_indices: torch.Tensor, bottom_n_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [num_nodes, d_model]
            top_n_indices: Indices for top neighbors [num_nodes, k]
            bottom_n_indices: Indices for bottom neighbors [num_nodes, k]
        """
        num_nodes = x.size(0)
        x_original = x.clone()
        
        # Create edge indices for different attention types
        self_edge_index = torch.arange(num_nodes, device=x.device)
        self_edge_index = torch.stack([self_edge_index, self_edge_index])
        
        top_edge_index = self._create_edge_index(top_n_indices, num_nodes)
        bottom_edge_index = self._create_edge_index(bottom_n_indices, num_nodes)
        
        # Apply GAT for each attention type
        self_out = self.self_gat(x, self_edge_index)
        top_out = self.top_gat(x, top_edge_index)
        bottom_out = self.bottom_gat(x, bottom_edge_index)
        
        # Calculate importance weights
        self_importance = self.importance_calc[0](self_out)
        top_importance = self.importance_calc[1](top_out)
        bottom_importance = self.importance_calc[2](bottom_out)
        
        # Combine importance weights
        importance_weights = torch.cat([self_importance, top_importance, bottom_importance], dim=-1)
        importance_weights = F.softmax(importance_weights, dim=-1)
        
        # Combine outputs
        output = (
            importance_weights[:, 0:1] * self_out +
            importance_weights[:, 1:2] * top_out +
            importance_weights[:, 2:3] * bottom_out
        )
        
        # Final processing
        output = self.activation(output)
        output = self.dropout(output)
        output = self.layer_norm(output + x_original)
        
        return output, importance_weights

class HGAIT(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, d_model: int, n_heads: int, mode: str, n_neighbors: float, n_layers: int = 2, dropout_rate: float = 0.4):
        super(HGAIT, self).__init__()
                
        # Time Mixing 레이어 정의
        self.time_mixer = TimeMixing(num_features=input_dim, sequence_length=sequence_length, d_model = d_model, mode = mode) # (노드 수, 변수 개수, 시퀀스 길이)
        
        # Variable Attention Layer를 n_layers만큼 쌓기
        self.variable_attention_layers = nn.ModuleList([
            VariableAttentionLayer(
                d_model=d_model,
                n_heads=n_heads,
                dropout_rate=dropout_rate
            ) for _ in range(n_layers)
        ])
        
        # Feature Importance 및 Attention 관련 블록 정의
        self.feature_importance_block = FeatureImportanceTokenizedBlock()
        
        self.gated_attention = GatedAttentionLayer(d_model = d_model, n_heads = n_heads, dropout_rate = dropout_rate)
        self.n_neighbors = n_neighbors
        
        # 예측기 정의
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2 * d_model, 1)  
        )
        
        # 최종 Layer Normalization
        self.layer_norm_final = nn.LayerNorm(d_model)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력 텐서 형상: [노드 수, 변수 개수, 시퀀스 길이] # 노드수 * percent
        # 해당 부분 최적화 필요
        
        ###### 상관관계 계산하는 부분 #######
        node_features = x[:, 4, :] # -5번째 변수가 Return이고, n_t개의 노드에서 L길이의 시퀀스 동안의 return 값들을 가져오는 것.
        mean_node_features = torch.mean(node_features, dim=-1, keepdim=True)
        centered_node_features = node_features - mean_node_features
        covariance_matrix = torch.matmul(centered_node_features, centered_node_features.transpose(0, 1)) / (node_features.shape[1] - 1)
        std_node_features = torch.sqrt(torch.sum(centered_node_features ** 2, dim=-1) / (node_features.shape[1] - 1))
        std_matrix = std_node_features.unsqueeze(1) * std_node_features.unsqueeze(0)
        correlation_matrix = covariance_matrix / std_matrix
        top_n_indices = torch.topk(correlation_matrix, k=int((x.shape[0])*(self.n_neighbors)), dim=-1).indices
        bottom_n_indices = torch.topk(-correlation_matrix, k=int(x.shape[0]*(self.n_neighbors)), dim=-1).indices

        # Time Mixing 레이어를 통과
        x = self.time_mixer(x)  # (노드 수, 변수 개수, 시퀀스 길이)
        
        # Variable Attention Layer를 통과
        for attention_layer in self.variable_attention_layers:
            x, feature_importance = attention_layer(x)

        # Feature Importance 계산.
        x = self.feature_importance_block(x, feature_importance)

        # Gated Attention 통과
        output, importance_weights = self.gated_attention(x, top_n_indices, bottom_n_indices)
                
        # 최종 예측 수행
        logits = self.predictor(self.layer_norm_final(output))
        
        return logits ,importance_weights
