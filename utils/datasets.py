import torch
from pathlib import Path
from abc import abstractmethod
import numpy as np
import tifffile
import multiprocessing
from utils import augmentations, experiment_manager, geofiles
from utils.experiment_manager import CfgNode


class AbstractPopDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.PATHS.DATASET)
        self.split = cfg.DATALOADER.SPLIT
        self.indices = [['Blue', 'Green', 'Red', 'NIR'].index(band) for band in cfg.DATALOADER.SPECTRAL_BANDS]
        self.rescale_factor = 3_000
        self.sites = cfg.DATALOADER.SITES
        self.all_samples = []
        for site in self.sites:
            site_samples = geofiles.load_json(self.root_path / f'samples_{site}.json')
            self.all_samples.extend(site_samples)
        self.labeled_samples = [s for s in self.all_samples if s['is_labeled']]

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def _get_planet_img(self, site: str) -> np.ndarray:
        file = self.root_path / f'planet_{site}.tif'
        img = tifffile.imread(file)
        img = np.clip(0, 1, img[:, :, self.indices] / self.rescale_factor)
        return img.astype(np.float32)

    def _load_planet_tile(self, site: str, i_tile: int, j_tile: int) -> np.ndarray:
        file = self.root_path / 'tiles' / site / f'planet_{site}_{i_tile:010d}_{j_tile:010d}.tif'
        tile = tifffile.imread(file)
        tile = np.clip(0, 1, tile[:, :, self.indices] / self.rescale_factor)
        return tile.astype(np.float32)

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


# dataset for urban extraction with building footprints
class PopDataset(AbstractPopDataset):

    def __init__(self, cfg: experiment_manager.CfgNode, run_type: str, no_augmentations: bool = False):
        super().__init__(cfg)

        self.run_type = run_type
        if self.split == 'random':
            random_numbers = np.random.rand(len(self.labeled_samples))
            if run_type == 'train':
                self.samples = [s for s, r in zip(self.labeled_samples, random_numbers) if r < 0.8]
            elif run_type == 'test':
                self.samples = [s for s, r in zip(self.labeled_samples, random_numbers) if r >= 0.8]
            elif run_type == 'unlabeled':
                self.samples = [s for s in self.all_samples if not s['is_labeled']]
            else:
                raise Exception('Unknown run type.')
        else:
            pass

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg.AUGMENTATION, no_augmentations)

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)
        self.length = len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        i_img, j_img, site = s['i_img'], s['j_img'], s['site']
        i_tile, j_tile = i_img // 1_000 * 1_000, j_img // 1_000 * 1_000
        tile = self._load_planet_tile(site, i_tile, j_tile)

        i_within_tile, j_within_tile = i_img % 1_000, j_img % 1_000
        tile = tile[i_within_tile:i_within_tile + 100, j_within_tile:j_within_tile + 100]

        x = self.transform(tile)
        if self.run_type == 'train' or self.run_type == 'test':
            y = float(s['pop'])
            assert not np.isnan(y) and y >= 0
        else:
            y = np.NaN

        item = {
            'x': x,
            'y': torch.tensor([y]),
            'i': i_img,
            'j': j_img,
            'id': s['id'],
            'site': s['site'],
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'