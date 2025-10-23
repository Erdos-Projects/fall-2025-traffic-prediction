# A collection of models for easy access

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.utils.data import TensorDataset, DataLoader

# Define consistent target transform
log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=False)

def make_preprocessor(features, scale=True):
    if scale:
        return ColumnTransformer([
            ('num', StandardScaler(with_mean=False), features)
        ], remainder='drop')
    else:
        return ColumnTransformer([
            ('num', 'passthrough', features)
        ], remainder='drop')

def make_model(preprocessor, regressor=LinearRegression(), transform_target=True):
    pipe = Pipeline([
        ('pre', preprocessor),
        ('reg', regressor)
    ])
    if transform_target:
        return TransformedTargetRegressor(regressor=pipe, transformer=log_transformer)
    else:
        return pipe
    
def evaluate_model(name, model, X_train, y_train, X_test, y_test, cv=False):
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start

    y_pred = model.predict(X_test)
    rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    if cv:
        cv_rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5))

    return {
        'model': name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'fit_time': fit_time,
        'cv_rmse_mean': cv_rmse.mean() if cv else 0,
        'cv_rmse_std': cv_rmse.std() if cv else 0
    }

def make_rf(**kwargs):
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        **kwargs
    )

def make_gbrt(**kwargs):
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        **kwargs
    )

def make_xgb(**kwargs):
    """Default XGBoost regressor for structured tabular data."""
    params = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.9,
        reg_lambda=0.3,
        reg_alpha=0.5,
        min_child_weight=20,
        n_jobs=-1,
        tree_method='hist',  # faster for large datasets
        objective='reg:squarederror',
    )
    params.update(kwargs)
    return XGBRegressor(**params)

# Build graph
def build_highway_graphs(tmc_order_dict):
    """
    Build separate line graphs for each travel direction.
    
    Args:
        tmc_order_dict: dict with direction name -> ordered list of TMC codes
                        e.g., {'NB': ['115+04177', '115+04178', ...],
                               'SB': ['115-04177', '115-04178', ...]}
    Returns:
        edge_index_dict: dict with same keys, each value = edge_index tensor (2 x E_dir)
        node_id_dict: dict mapping direction -> {tmc_code: node_id_within_dir}
    """
    edge_index_dict = {}
    node_id_dict = {}

    for direction, tmc_list in tmc_order_dict.items():
        G = nx.Graph()
        for i in range(len(tmc_list) - 1):
            G.add_edge(i, i + 1)
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        edge_index_dict[direction] = edge_index
        node_id_dict[direction] = {tmc: i for i, tmc in enumerate(tmc_list)}

    return edge_index_dict, node_id_dict


def dataframe_to_tensors_by_direction(X_full, feature_cols, target_col, tmc_order_dict):
    """
    Convert a multi-indexed DataFrame (tmc_code, time_bin) into
    per-direction tensors for spatio-temporal GNN training.

    Args:
        X_full : DataFrame with MultiIndex ['tmc_code', 'time_bin']
        feature_cols : list of feature column names
        target_col : target column name (e.g. 'tt_per_mile')
        tmc_order_dict : dict of direction -> ordered list of TMCs
    
    Returns:
        tensors_dict : {
            direction: {
                'X': [T, N, F] tensor,
                'Y': [T, N] tensor,
                'tmc_order': list of TMCs,
                'time_index': list of time bins
            }, ...
        }
    """
    X_full = X_full.sort_index(level=['time_bin', 'tmc_code'])
    time_index = X_full.index.get_level_values('time_bin').unique().tolist()
    T = len(time_index)
    F = len(feature_cols)

    tensors_dict = {}

    for direction, tmc_order_list in tmc_order_dict.items():
        N = len(tmc_order_list)
        X = np.zeros((T, N, F), dtype=np.float32)
        Y = np.zeros((T, N), dtype=np.float32)

        for ti, t in enumerate(time_index):
            df_t = X_full.xs(t, level='time_bin').reindex(tmc_order_list)
            X[ti] = df_t[feature_cols].to_numpy()
            Y[ti] = df_t[target_col].to_numpy()

        tensors_dict[direction] = {
            'X': torch.tensor(X),
            'Y': torch.tensor(Y),
            'tmc_order': tmc_order_list,
            'time_index': time_index
        }

    return tensors_dict

def build_pyg_data_dict(tensors_dict, edge_index_dict):
    """
    Wrap per-direction tensors and graph edges into PyG Data objects.
    
    Args:
        tensors_dict: output of dataframe_to_tensors_by_direction()
        edge_index_dict: output of build_highway_graphs()
    
    Returns:
        data_dict: dict(direction -> PyG Data)
    """
    data_dict = {}
    for direction in tensors_dict.keys():
        X = tensors_dict[direction]['X']     # [T, N, F]
        Y = tensors_dict[direction]['Y']     # [T, N]
        edge_index = edge_index_dict[direction]  # [2, E]

        # We’ll store temporal tensors directly in the object for convenience.
        data = Data(edge_index=edge_index)
        data.X = X
        data.Y = Y
        data.tmc_order = tensors_dict[direction]['tmc_order']
        data.time_index = tensors_dict[direction]['time_index']
        data_dict[direction] = data

    return data_dict


class LogSpaceMSE(nn.Module):
    def forward(self, y_true, y_pred):
        eps = 1e-6
        y_true = torch.clamp(y_true, min=eps)
        y_pred = torch.clamp(y_pred, min=eps)
        return ((torch.log(y_true) - torch.log(y_pred))**2).mean()
    
class GCN_LSTM(nn.Module):
    def __init__(self, in_features, gcn_hidden, lstm_hidden, out_features, edge_index, dropout=0.2):
        super().__init__()
        self.register_buffer('edge_index', edge_index)

        # --- Input normalization layer (acts like Keras Normalization)
        self.input_norm = nn.BatchNorm1d(in_features, affine=False, track_running_stats=True)

        # --- Spatial and temporal blocks
        self.gcn = GCNConv(in_features, gcn_hidden)
        self.bn_gcn = nn.BatchNorm1d(gcn_hidden)   # stabilize GCN output
        self.lstm = nn.LSTM(gcn_hidden, lstm_hidden, batch_first=True)

        # --- Head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, out_features)

    def _build_batched_edge_index(self, B, N):
        base = self.edge_index
        E = base.size(1)
        offsets = torch.arange(B, device=base.device).view(-1, 1, 1) * N
        ei = base.view(1, 2, E) + offsets
        ei = ei.permute(1, 0, 2).reshape(2, B * E)
        return ei

    def forward(self, X_seq):
        # X_seq: [B, T, N, F]
        B, T, N, F = X_seq.shape
        batched_edge_index = self._build_batched_edge_index(B, N)
        spatial_out = []

        # --- Normalize inputs feature-wise before GCN (like StandardScaler)
        # Flatten across batch, time, nodes → apply BN on feature dimension
        X_seq_flat = X_seq.reshape(-1, F)
        X_seq_flat = self.input_norm(X_seq_flat)
        X_seq = X_seq_flat.view(B, T, N, F)

        for t in range(T):
            x_t = X_seq[:, t].reshape(B * N, F)
            g = self.gcn(x_t, batched_edge_index)
            g = self.bn_gcn(g)
            g = torch.relu(g).view(B, N, -1)
            spatial_out.append(g)

        H = torch.stack(spatial_out, dim=1)  # [B, T, N, gcn_hidden]
        H = H.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        lstm_out, _ = self.lstm(H)
        last = self.dropout(lstm_out[:, -1])
        pred = self.fc2(self.act(self.fc1(last))).view(B, N, -1)
        return pred

def train_gcn_lstm(model, data, seq_len=6, epochs=100, lr=1e-3,
                      batch_size=128, val_frac=0.1, clip=1.0,
                      patience_limit=15, device='cpu'):
    """
    Trains a GCN-LSTM model with in-model normalization (no external scaler).

    Args:
        model : nn.Module (GCN_LSTM_v4)
        data : PyG Data object with .X [T,N,F], .Y [T,N]
        seq_len : lookback window
        val_frac : fraction of train set for validation
        patience_limit : early stopping patience
    """
    if device == 'default':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X = data.X.to(device)   # [T, N, F]
    Y = data.Y.to(device)   # [T, N]
    T_total = X.size(0)
    n_train = int(T_total * (1 - val_frac))

    # --- Build sliding windows ---
    def make_windows(X_tensor, Y_tensor):
        X_list, Y_list = [], []
        for t in range(seq_len, X_tensor.size(0)):
            X_list.append(X_tensor[t - seq_len:t].unsqueeze(0))
            Y_list.append(Y_tensor[t].unsqueeze(0).unsqueeze(-1))
        return torch.cat(X_list), torch.cat(Y_list)

    X_train, Y_train = make_windows(X[:n_train], Y[:n_train])
    X_val, Y_val     = make_windows(X[n_train - seq_len:], Y[n_train - seq_len:])

    # --- DataLoaders (shuffle training) ---
    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float('inf')
    patience = 0
    best_state = None

    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    for epoch in range(epochs):
        # ----- TRAIN -----
        model.train()  # enables BatchNorm & Dropout updates
        total_loss = 0.0

        for Xb, Yb in train_loader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss = loss_fn(preds, Yb)
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # ----- VALIDATION -----
        model.eval()   # freeze BN statistics (no running mean updates)
        val_loss = 0.0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                preds = model(Xb)
                val_loss += loss_fn(preds, Yb).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        # ----- Early stopping -----
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def predict_sequence(model, data, seq_len=24, horizon=1, device='cpu'):
    if device == 'default':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    X = data.X.to(device)
    T = X.size(0)
    preds = []
    with torch.no_grad():
        for t in range(seq_len, T - horizon + 1):
            x_seq = X[t-seq_len:t].unsqueeze(0)
            y_hat = model(x_seq).squeeze(0)  # [N, out_features]
            preds.append(y_hat.cpu())
    return torch.stack(preds)  # [T-seq_len-horizon+1, N, out_features]


# ==== PLOTTING ========
def plot_heatmap_travel_time(
    preds,                      # np.ndarray or torch.Tensor [T, N]
    time_index,                 # iterable of timestamps (aligned with preds)
    tmc_order,                  # list of TMC codes or ordered segments
    title="Predicted Travel Time Heatmap",
    cmap='viridis',
    vmin=None,
    vmax=None,
    center=None,
    freq=None,                  # Optional resample frequency (e.g., '1H')
    agg='mean'
):
    """
    Plot a spatio-temporal heatmap for travel time predictions from any model.

    Args:
        preds (array): [T, N] predictions (timesteps × TMCs)
        time_index (array-like): time labels for each row in preds
        tmc_order (array-like): ordered TMC codes (columns)
        title (str): plot title
        cmap (str): colormap
        vmin, vmax (float): color scale bounds
        freq (str): optional pandas resample frequency (e.g. '1H')
        agg (str): aggregation method if resampling
    """
    # Convert tensors → NumPy
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    elif isinstance(preds, list):
        preds = np.array(preds)

    # Build DataFrame [time × tmc]
    df_heat = pd.DataFrame(preds, index=pd.to_datetime(time_index), columns=tmc_order)

    # # Optional temporal aggregation (resample)
    # if freq is not None and isinstance(df_heat.index, pd.DatetimeIndex):
    #     df_heat = df_heat.resample(freq).agg(agg)

    # ---- Plot heatmap ----
    plt.figure(figsize=(14, 6))
    ax = sns.heatmap(
        df_heat.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cbar_kws={'label': 'Travel Time per Mile (seconds)'},
        mask=df_heat.T.isna()
    )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("TMC (ordered index)")

    # --- Format x-axis tick labels (drop last 10 characters) ---
    xticklabels = [label.get_text()[:-10] for label in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels, rotation=90, ha='center')

    plt.tight_layout()
    plt.show()

    return df_heat