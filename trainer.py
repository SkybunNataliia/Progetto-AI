import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from models.cnn1d_model import CNN1DModel
from models.gru_model import GRUModel
from models.lstm_model import LSTMModel
from utils.data_loader import get_dataloaders

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._get_model().to(self.device)

        batch_size = cfg.hyper_parameters.batch_size
        seq_len = cfg.hyper_parameters.window_size

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            cfg.io.training_file,
            cfg.io.validation_file,
            cfg.io.test_file,
            batch_size=batch_size,
            seq_len=seq_len
        )

        self.out_root = Path(cfg.io.out_folder)
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.last_model_path = self.out_root / "last_model.pth"
        self.best_model_path = self.out_root / "best_model.pth"
        
        if self.cfg.train_parameters.reload_last_model:
            try:
                self.model.load_state_dict(torch.load(self.last_model_outpath_sd))
                print('Last model state_dict successfully reloaded.')
            except Exception as e:
                print(f'Cannot reload last model state_dict: {e}')


        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=cfg.hyper_parameters.learning_rate,
            momentum=cfg.hyper_parameters.momentum
        )

        self.epochs = cfg.hyper_parameters.epochs
        self.step_monitor = cfg.train_parameters.step_monitor
        self.loss_target = cfg.train_parameters.loss_target

        self.early_stop_start = cfg.early_stop_parameters.start_epoch
        self.early_stop_eval_freq = cfg.early_stop_parameters.loss_evaluation_epochs
        self.early_stop_patience = cfg.early_stop_parameters.patience
        self.early_stop_improve_rate = cfg.early_stop_parameters.improvement_rate
    
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
            for i, (inputs, targets) in enumerate(self.train_loader, 1):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if self.step_monitor > 0 and i % self.step_monitor == 0:
                    avg_loss = running_loss / self.step_monitor
                    print(f"Epoch {epoch} Step {i} - Loss: {avg_loss:.6f}")
                    running_loss = 0.0

            val_loss = self.validate()
            print(f"Epoch {epoch} Validation Loss: {val_loss:.6f}")

            if val_loss + self.early_stop_improve_rate < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Best model saved at epoch {epoch} with val loss {val_loss:.6f}")
            else:
                epochs_no_improve += 1
                if epoch >= self.early_stop_start and epochs_no_improve >= self.early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            if val_loss <= self.loss_target:
                print(f"Loss target {self.loss_target} reached at epoch {epoch}. Stopping training.")
                break

            torch.save(self.model.state_dict(), self.last_model_path)
            
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets)
                total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(self.val_loader.dataset)
        return avg_loss
    
    