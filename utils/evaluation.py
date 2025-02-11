import torch
from torch.utils import data as torch_data
import wandb
from utils import datasets, networks, experiment_manager
import numpy as np
from scipy import stats
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RegressionEvaluation(object):
    def __init__(self, name: str = None):
        self.name = name
        self.predictions = []
        self.labels = []

    def add_sample_numpy(self, pred: np.ndarray, label: np.ndarray):
        self.predictions.extend(pred.flatten())
        self.labels.extend(label.flatten())

    def add_sample_torch(self, pred: torch.tensor, label: torch.tensor):
        pred = pred.float().detach().cpu().numpy()
        label = label.float().detach().cpu().numpy()
        self.add_sample_numpy(pred, label)

    def reset(self):
        self.predictions = []
        self.labels = []

    def root_mean_square_error(self) -> float:
        return np.sqrt(np.sum(np.square(np.array(self.predictions) - np.array(self.labels))) / len(self.labels))

    def r_square(self) -> float:
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.labels, self.predictions)
        return r_value


def model_evaluation(net: networks.PopulationNet, cfg: experiment_manager.CfgNode, run_type: str, epoch: float):
    net.to(device)
    net.eval()

    measurer = RegressionEvaluation()
    dataset = datasets.PopDataset(cfg, run_type, no_augmentations=True)
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            img = batch['x'].to(device)
            label = batch['y'].to(device)
            pred = net(img)
            measurer.add_sample_torch(pred, label)

    # assessment
    rmse = measurer.root_mean_square_error()
    print(f'RMSE {run_type} {rmse:.3f}')
    wandb.log({
        f'{run_type} rmse': rmse,
        'epoch': epoch,
    })

    return rmse
