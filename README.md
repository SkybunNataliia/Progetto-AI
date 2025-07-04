# Deep Learning Forecasting Project

Previsione multivariata su serie temporali tramite modelli LSTM, GRU e CNN sviluppati con PyTorch.

---

## Setup del Progetto con Miniconda

### 1. Prerequisiti

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installato
- Python **3.11**
- `git` (opzionale, per il clone del progetto)

### 2. Creazione dell'ambiente Conda

Esegui i seguenti comandi per creare ed attivare l'ambiente virtuale:

```bash
# Clona il progetto (opzionale)
git clone https://github.com/SkybunNataliia/Progetto-AI.git
cd progetto-forecasting

# Crea l’ambiente da environment.yml
conda env create -f environment.yml

# Attiva l’ambiente
conda activate dl-forecasting
```

### 3. Logging di Tensorboard

Per visualizzare i log delle metriche effettuati con Tensorboard eseguire il comando:

```bash
tensorboard --logdir_spec=lstm:experiments/lstm/v4/runs,gru:experiments/gru/v7/runs,cnn:experiments/cnn/v4/runs
```