import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
from pathlib import Path
from models.cnn1d_model import CNN1DModel
from models.gru_model import GRUModel
from models.lstm_model import LSTMModel
from utils.data_loader import get_dataloaders
from utils.metrics import evaluate_all
from utils.seed_utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from utils.experiment_utils import create_experiment_dir, get_next_version

class Trainer:
    def __init__(self, cfg, scaler_target=None, version=None, base_exp_path="experiments"):
        
        set_seed(42)
        self.cfg = cfg
        
        model_name = cfg.train_parameters.network_type.lower()
        model_dir = Path(base_exp_path) / model_name
        
        if version is None:
            version = get_next_version(model_dir)
            
        self.exp_dir = create_experiment_dir(base_exp_path, model_name, version, "./config/config.json")
        
        self.writer = SummaryWriter(log_dir=str(self.exp_dir / "runs"))
        
        self.out_root = self.exp_dir
        self.out_root.mkdir(parents=True, exist_ok=True)
        
        self.results_log = {
            "model": model_name,
            "version": version,
            "hyperparameters": vars(self.cfg.hyper_parameters),
            "training_log": [],
            "validation_loss_per_epoch": [],
            "early_stopping_epoch": None,
            "test_results": {}
        }

        self.model = self._get_model()
        
        sample_input = torch.randn(1, self.cfg.hyper_parameters.window_size, self._input_size())
        self.writer.add_graph(self.model, sample_input)
        
        self.scaler_target = scaler_target

        batch_size = cfg.hyper_parameters.batch_size
        seq_len = cfg.hyper_parameters.window_size

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            cfg.io.training_file,
            cfg.io.validation_file,
            cfg.io.test_file,
            batch_size=batch_size,
            seq_len=seq_len
        )
        
        self.last_model_path = self.out_root / f"{model_name}_last_model.pth"
        self.best_model_path = self.out_root / f"{model_name}_best_model.pth"
        
        if self.cfg.train_parameters.reload_last_model:
            try:
                self.model.load_state_dict(torch.load(self.last_model_path))
                print('Last model state_dict successfully reloaded.')
            except Exception as e:
                print(f'Cannot reload last model state_dict: {e}')

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=cfg.hyper_parameters.learning_rate, 
            weight_decay=1e-4
        )

        self.epochs = cfg.hyper_parameters.epochs
        self.step_monitor = cfg.train_parameters.step_monitor
        self.loss_target = cfg.train_parameters.loss_target

        self.early_stop_start = cfg.early_stop_parameters.start_epoch
        self.early_stop_eval_freq = cfg.early_stop_parameters.loss_evaluation_epochs
        self.early_stop_patience = cfg.early_stop_parameters.patience
        self.early_stop_improve_rate = cfg.early_stop_parameters.improvement_rate
    
    def _log(self, message: str):
        print(message)
        self.results_log["training_log"].append(message)
    
    def _input_size(self):
        df = pd.read_csv(self.cfg.io.training_file)
        return df.shape[1] - 1
    
    def _get_model(self):
        net_type = self.cfg.train_parameters.network_type.lower()
        if net_type == "lstm":
            return LSTMModel(
                input_size=self._input_size(),
                hidden_size=self.cfg.rnn_parameters.hidden_size,
                num_layers=self.cfg.rnn_parameters.num_layers,
            )
        elif net_type == "gru":
            return GRUModel(
                input_size=self._input_size(),
                hidden_size=self.cfg.rnn_parameters.hidden_size,
                num_layers=self.cfg.rnn_parameters.num_layers,
            )
        elif net_type == "cnn":
            return CNN1DModel(
                input_size=self._input_size(),
                num_filters=self.cfg.cnn_parameters.num_filters,
                kernel_size=self.cfg.cnn_parameters.kernel_size,
                stride=self.cfg.cnn_parameters.stride,
                padding=self.cfg.cnn_parameters.padding,
            )
        else:
            raise ValueError(f"Unsupported network type: {net_type}")

    def train(self):
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            
            for i, (inputs, targets) in enumerate(self.train_loader, 0):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()

                if self.step_monitor > 0 and i % self.step_monitor == 0:
                    avg_loss = running_loss / self.step_monitor
                    self._log(f"Epoch {epoch} Step {i+1} - Loss: {avg_loss:.6f}")
                    
            epoch_train_loss = running_loss / len(self.train_loader)
            val_loss = self.validate()
            
            print(f"[Validation Loss]: {val_loss:.6f}")
            self.results_log["validation_loss_per_epoch"].append(val_loss)

            self.writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            # Logga parametri e gradienti
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.writer.add_histogram(f"{name}/grad", param.grad, epoch)
                    self.writer.add_histogram(f"{name}/data", param.data, epoch)

            # Early stop
            if val_loss + self.early_stop_improve_rate < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"[{self.cfg.train_parameters.network_type.upper()}] Best model saved at epoch {epoch} with val loss {val_loss:.6f}")
            else:
                epochs_no_improve += 1
                if epoch >= self.early_stop_start and epochs_no_improve >= self.early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    self.results_log["early_stopping_epoch"] = epoch
                    break

            if val_loss <= self.loss_target:
                self._log(f"Loss target {self.loss_target} reached at epoch {epoch}. Stopping training.")
                break

            torch.save(self.model.state_dict(), self.last_model_path)
            
            results_path = self.out_root / "results.json"
            with open(results_path, "w") as f:
                json.dump(self.results_log, f, indent=4)
            
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                count += 1

        return total_loss / count if count > 0 else None
    
    def evaluate(self, data_loader):
        self.model.eval()
        preds = []
        targets_list = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.model(inputs)
                preds.extend(outputs.cpu().detach().numpy().tolist())
                targets_list.extend(targets.cpu().detach().numpy().tolist())
                
        if self.scaler_target:
            preds_original = self.scaler_target.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
            targets_original = self.scaler_target.inverse_transform(np.array(targets_list).reshape(-1,1)).flatten()
        
        metrics = evaluate_all(preds_original, targets_original)
        
        return metrics, preds, targets_list
    
    def test(self, use_current_model: bool = False, print_loss=True):
        if use_current_model:
            model = self.model
        else:
            model = self._get_model()
            try:
                model.load_state_dict(torch.load(self.best_model_path, map_location='cpu'))
                print("Best model loaded for testing.")
            except Exception as e:
                print(f'Error loading model state_dict: {repr(e)}')
                model = self.model

        self.model = model
        
        metrics, preds, targets = self.evaluate(self.test_loader)

        if print_loss:
            print(f"Test MSE: {metrics['MSE']:.6f}")
            print(f"Test MAPE: {metrics['MAPE']:.2f}%")
            
        self.results_log["test_results"] = metrics
        
        with open(self.out_root / "results.json", "w") as f:
            json.dump(self.results_log, f, indent=4)
        
        # Logging TensorBoard
        test_loss = metrics.get('MSE', None)
        if test_loss is not None:
            self.writer.add_scalar('Loss/test', test_loss, 0)
            
        return metrics, preds, targets