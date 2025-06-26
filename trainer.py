import torch
import torch.nn as nn
import torch.optim as optim
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

