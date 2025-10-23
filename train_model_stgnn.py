"""
train_model_stgnn.py

One-click, end-to-end training and inference script for a Spatio-Temporal GNN (GCN+LSTM)
on the I-10 corridor dataset. It loads feature engineering output (X_full_1h.parquet),
builds per-direction spatio-temporal tensors, defines a GCN-LSTM model, trains with
validation and early stopping, runs inference, and saves artifacts:

- model_weights.pth: Trained PyTorch weights
- data_object.pt: PyG Data object with tensors and metadata
- predictions.npz: Predictions, ground truth, time index, TMC order, seq_len

Examples:
  python train_model_stgnn.py --direction WB --save-dir models/gcn/gcn_lstm_i10_wb
  python train_model_stgnn.py --data-path database/i10-broadway --file X_full_1h.parquet --direction EB
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ---------------------------
# Utilities & configuration
# ---------------------------

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_device(user_choice: str = "auto") -> torch.device:
    """Resolve device selection.

    user_choice: "auto" | "cpu" | "cuda" | "mps"
    """
    if user_choice and user_choice.lower() != "auto":
        return torch.device(user_choice)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------
# Graph and tensor builders
# ---------------------------

def build_highway_graphs(tmc_order_dict: Dict[str, List[str]]):
    """
    Build separate line graphs for each travel direction.

    Args:
        tmc_order_dict: dict with direction name -> ordered list of TMC codes

    Returns:
        edge_index_dict: dict with same keys, each value = edge_index tensor (2 x E_dir)
        node_id_dict: dict mapping direction -> {tmc_code: node_id_within_dir}
    """
    edge_index_dict = {}
    node_id_dict = {}

    for direction, tmc_list in tmc_order_dict.items():
        # Line graph: connect i -> i+1
        edges = []
        for i in range(len(tmc_list) - 1):
            edges.append((i, i + 1))
        if not edges:
            raise ValueError(f"Direction {direction} has <2 nodes; cannot form edges.")
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index_dict[direction] = edge_index
        node_id_dict[direction] = {tmc: i for i, tmc in enumerate(tmc_list)}

    return edge_index_dict, node_id_dict


def dataframe_to_tensors_by_direction(
    X_full: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    tmc_order_dict: Dict[str, List[str]],
) -> Dict[str, Dict[str, torch.Tensor | list]]:
    """
    Convert a multi-indexed DataFrame (tmc_code, time_bin) into per-direction tensors
    for spatio-temporal GNN training.

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
    # Ensure expected index
    if not isinstance(X_full.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex (tmc_code, time_bin). Got single Index.")
    if set(X_full.index.names) != {"tmc_code", "time_bin"}:
        raise ValueError(f"Index level names mismatch. Found {X_full.index.names}")

    # Sort by time, then tmc for stable layout
    X_full = X_full.sort_index(level=["time_bin", "tmc_code"]) 
    time_index = X_full.index.get_level_values("time_bin").unique().tolist()
    T = len(time_index)
    F = len(feature_cols)

    tensors_dict: Dict[str, Dict[str, torch.Tensor | list]] = {}

    for direction, tmc_order_list in tmc_order_dict.items():
        N = len(tmc_order_list)
        X = np.zeros((T, N, F), dtype=np.float32)
        Y = np.zeros((T, N), dtype=np.float32)

        for ti, t in enumerate(time_index):
            df_t = X_full.xs(t, level="time_bin").reindex(tmc_order_list)
            X[ti] = df_t[feature_cols].to_numpy()
            Y[ti] = df_t[target_col].to_numpy()

        tensors_dict[direction] = {
            "X": torch.tensor(X),
            "Y": torch.tensor(Y),
            "tmc_order": tmc_order_list,
            "time_index": time_index,
        }

    return tensors_dict


def build_pyg_data_dict(
    tensors_dict: Dict[str, Dict[str, torch.Tensor | list]],
    edge_index_dict: Dict[str, torch.Tensor],
) -> Dict[str, Data]:
    """Wrap per-direction tensors and graph edges into PyG Data objects."""
    data_dict: Dict[str, Data] = {}
    for direction in tensors_dict.keys():
        X = tensors_dict[direction]["X"]  # [T, N, F]
        Y = tensors_dict[direction]["Y"]  # [T, N]
        edge_index = edge_index_dict[direction]  # [2, E]

        data = Data(edge_index=edge_index)
        data.X = X
        data.Y = Y
        data.tmc_order = tensors_dict[direction]["tmc_order"]
        data.time_index = tensors_dict[direction]["time_index"]
        data_dict[direction] = data

    return data_dict


# ---------------------------
# Model definition
# ---------------------------

class GCN_LSTM(nn.Module):
    def __init__(
        self,
        in_features: int,
        gcn_hidden: int,
        lstm_hidden: int,
        out_features: int,
        edge_index: torch.Tensor,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.register_buffer("edge_index", edge_index)

        # Input normalization akin to Keras Normalization (feature-wise BN)
        self.input_norm = nn.BatchNorm1d(in_features, affine=False, track_running_stats=True)

        # Spatial and temporal blocks
        self.gcn = GCNConv(in_features, gcn_hidden)
        self.bn_gcn = nn.BatchNorm1d(gcn_hidden)
        self.lstm = nn.LSTM(gcn_hidden, lstm_hidden, batch_first=True)

        # Head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, out_features)

    def _build_batched_edge_index(self, B: int, N: int) -> torch.Tensor:
        base = self.edge_index
        E = base.size(1)
        offsets = torch.arange(B, device=base.device).view(-1, 1, 1) * N
        ei = base.view(1, 2, E) + offsets
        ei = ei.permute(1, 0, 2).reshape(2, B * E)
        return ei

    def forward(self, X_seq: torch.Tensor) -> torch.Tensor:
        # X_seq: [B, T, N, F]
        B, T, N, F = X_seq.shape
        batched_edge_index = self._build_batched_edge_index(B, N)
        spatial_out: List[torch.Tensor] = []

        # Normalize inputs feature-wise before GCN
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


# ---------------------------
# Training & inference
# ---------------------------

def train_gcn_lstm(
    model: nn.Module,
    data: Data,
    seq_len: int = 24,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    val_frac: float = 0.2,
    clip: float | None = 1.0,
    patience_limit: int = 10,
    device: torch.device | str = "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a GCN-LSTM with early stopping. Prints train/val loss each epoch.

    Returns:
        model: Trained model with best weights restored
        history: Dict with keys {train_loss, val_loss, best_val, best_epoch, epochs_trained}
    """
    device = auto_device(device if isinstance(device, str) else str(device))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X = data.X.to(device)   # [T, N, F]
    Y = data.Y.to(device)   # [T, N]
    T_total = X.size(0)
    n_train = int(T_total * (1 - val_frac))

    # --- Build sliding windows ---
    def make_windows(X_tensor: torch.Tensor, Y_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X_list, Y_list = [], []
        for t in range(seq_len, X_tensor.size(0)):
            X_list.append(X_tensor[t - seq_len:t].unsqueeze(0))
            Y_list.append(Y_tensor[t].unsqueeze(0).unsqueeze(-1))
        return torch.cat(X_list), torch.cat(Y_list)

    X_train, Y_train = make_windows(X[:n_train], Y[:n_train])
    X_val, Y_val     = make_windows(X[n_train - seq_len:], Y[n_train - seq_len:])

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, Y_val),     batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    patience = 0
    best_state = None
    best_epoch = -1
    train_hist: List[float] = []
    val_hist: List[float] = []

    print(f"Device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

    for epoch in range(epochs):
        # ----- TRAIN -----
        model.train()
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
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                preds = model(Xb)
                val_loss += loss_fn(preds, Yb).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}")
        train_hist.append(float(train_loss))
        val_hist.append(float(val_loss))

        # ----- Early stopping -----
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history: Dict[str, Any] = {
        "train_loss": train_hist,
        "val_loss": val_hist,
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(len(train_hist)),
        "val_frac": float(val_frac),
        "seq_len": int(seq_len),
    }

    return model, history


def predict_sequence(
    model: nn.Module,
    data: Data,
    seq_len: int = 24,
    horizon: int = 1,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    model.eval()
    device = auto_device(device if isinstance(device, str) else str(device))
    X = data.X.to(device)
    T = X.size(0)
    preds = []
    with torch.no_grad():
        for t in range(seq_len, T - horizon + 1):
            x_seq = X[t - seq_len:t].unsqueeze(0)
            y_hat = model(x_seq).squeeze(0)  # [N, out_features]
            preds.append(y_hat.cpu())
    return torch.stack(preds)  # [T-seq_len-horizon+1, N, out_features]


# ---------------------------
# CLI main
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a GCN-LSTM ST-GNN on I-10 data")
    p.add_argument("--data-path", type=Path, default=Path("database/i10-broadway"), help="Directory containing X_full_1h.parquet")
    p.add_argument("--file", type=str, default="X_full_1h.parquet", help="Parquet filename with MultiIndex (tmc_code, time_bin)")
    p.add_argument("--direction", type=str, default="WB", choices=["WB", "EB"], help="Travel direction to train on")

    p.add_argument("--seq-len", type=int, default=24, help="Lookback window length")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    p.add_argument("--clip", type=float, default=1.0, help="Gradient clip norm (0/None to disable)")
    p.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|cuda|mps")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    p.add_argument("--gcn-hidden", type=int, default=64, help="GCN hidden features")
    p.add_argument("--lstm-hidden", type=int, default=128, help="LSTM hidden size")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save model and outputs (default: models/gcn/gcn_lstm_i10_<direction>)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)

    # Feature specification (mirrors notebook)
    time_features = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "hour_of_week_sin", "hour_of_week_cos", "is_weekend"
    ]
    evt_features = ["evt_cat_unplanned", "evt_cat_planned"]
    lag_features = ["lag1_tt_per_mile", "lag2_tt_per_mile", "lag3_tt_per_mile"]
    tmc_features = ["miles", "reference_speed", "curve", "onramp", "offramp"]
    FEATURE_COLS = time_features + evt_features + lag_features + tmc_features
    TARGET_COL = "tt_per_mile"

    # TMC order (WB/EB)
    tmc_order_dict = {
        "WB": [
            "115P04188", "115+04188", "115P04187", "115+04187", "115P04186", "115+04186",
            "115P04185", "115+04185", "115P04184", "115+04184", "115P04183", "115+04183",
            "115P04182", "115+04182", "115P04181", "115+04181", "115P04180", "115+04180",
            "115P04179", "115+04179", "115P04178", "115+04178", "115P04177", "115+04177",
            "115P05165"
        ],
        "EB": [
            "115N04188", "115-04187", "115N04187", "115-04186", "115N04186", "115-04185",
            "115N04185", "115-04184", "115N04184", "115-04183", "115N04183", "115-04182",
            "115N04182", "115-04181", "115N04181", "115-04180", "115N04180", "115-04179",
            "115N04179", "115-04178", "115N04178", "115-04177", "115N04177", "115-05165",
            "115N05165"
        ],
    }

    # Load engineered features
    parquet_path = args.data_path / args.file
    if not parquet_path.exists():
        raise FileNotFoundError(f"Could not find parquet at {parquet_path}.\n"
                                f"Ensure feature engineering step produced X_full_1h.parquet.")

    print(f"Loading: {parquet_path}")
    X_full = pd.read_parquet(parquet_path)

    # Build graph + tensors
    edge_index_dict, node_id_dict = build_highway_graphs(tmc_order_dict)
    tensors_dict = dataframe_to_tensors_by_direction(X_full, FEATURE_COLS, TARGET_COL, tmc_order_dict)
    data_dict = build_pyg_data_dict(tensors_dict, edge_index_dict)

    direction = args.direction.upper()
    if direction not in data_dict:
        raise ValueError(f"Direction {direction} not found. Available: {list(data_dict.keys())}")
    data = data_dict[direction]
    edge_index = edge_index_dict[direction]

    # Initialize model
    in_features = data.X.shape[-1]
    model = GCN_LSTM(
        in_features=in_features,
        gcn_hidden=args.gcn_hidden,
        lstm_hidden=args.lstm_hidden,
        out_features=1,
        edge_index=edge_index,
        dropout=args.dropout,
    )

    # Train
    device = auto_device(args.device)
    model, history = train_gcn_lstm(
        model,
        data,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        patience_limit=args.patience,
        clip=(None if args.clip in (0, None) else args.clip),
        device=device,
    )

    # Predict
    preds = predict_sequence(model, data, seq_len=args.seq_len, device=device).squeeze(-1)  # [T_eval, N]

    # Save artifacts
    # Resolve save directory: default matches direction
    save_dir: Path = args.save_dir or Path(f"models/gcn/gcn_lstm_i10_{direction.lower()}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Attach training history to the Data object for later retrieval
    try:
        data.train_loss_history = torch.tensor(history.get("train_loss", []), dtype=torch.float32)
        data.val_loss_history = torch.tensor(history.get("val_loss", []), dtype=torch.float32)
        data.best_val = torch.tensor(history.get("best_val", float("nan")), dtype=torch.float32)
        data.best_epoch = torch.tensor(history.get("best_epoch", -1), dtype=torch.int32)
        data.seq_len = torch.tensor(args.seq_len, dtype=torch.int32)
    except Exception:
        pass

    torch.save(model.state_dict(), save_dir / "model_weights.pth")
    torch.save(data, save_dir / "data_object.pt")
    np.savez_compressed(
        save_dir / "predictions.npz",
        preds=preds.detach().cpu().numpy(),
        Y=data.Y.cpu().numpy(),
        tmc_order=np.array(data.tmc_order),
        time_index=np.array(data.time_index),
        seq_len=np.array(args.seq_len),
        direction=np.array(direction),
    )

    # Also persist a lightweight metrics.json for comparison scripts
    try:
        import json
        metrics = {
            "best_val": history.get("best_val"),
            "best_epoch": history.get("best_epoch"),
            "epochs_trained": history.get("epochs_trained"),
            "val_frac": history.get("val_frac"),
            "seq_len": history.get("seq_len"),
            # store full histories for convenience
            "train_loss": history.get("train_loss"),
            "val_loss": history.get("val_loss"),
        }
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f)
    except Exception:
        pass

    print("\n✅ Training complete.")
    print(f"Saved to: {save_dir}")
    print("- model_weights.pth\n- data_object.pt\n- predictions.npz")


if __name__ == "__main__":
    # Allow MKL duplicate in some environments to avoid import warnings
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
