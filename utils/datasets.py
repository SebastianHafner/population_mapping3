import torch
from pathlib import Path
from abc import abstractmethod
import numpy as np
import multiprocessing
from utils import augmentations, experiment_manager, geofiles
from utils.experiment_manager import CfgNode
from affine import Affine


class AbstractPopDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: experiment_manager.CfgNode):
        super().__init__()
        self.cfg = cfg
        self.root_path = Path(cfg.PATHS.DATASET)
        self.split = cfg.DATALOADER.SPLIT
        self.indices = [['Blue', 'Green', 'Red', 'NIR'].index(band) for band in cfg.DATALOADER.SPECTRAL_BANDS]
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
        img, geo_transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.indices] / 3_000
        return img.astype(np.float32)

    def _get_unit_pop(self, unit_nr: int, year: int) -> int:
        return int(self.metadata['census'][str(unit_nr)][f'pop{year}'])

    def _get_unit_popgrowth(self, unit_nr: int) -> int:
        return int(self.metadata['census'][str(unit_nr)]['difference'])

    def _get_unit_split(self, unit_nr: int) -> str:
        return str(self.metadata['census'][str(unit_nr)][f'split'])

    def _get_pop_label(self, site: str, year: int, i: int, j: int) -> float:
        for s in self.metadata:
            if s['site'] == site and s['year'] == year and s['i'] == i and s['j'] == j:
                return float(s['pop'])
        raise Exception('sample not found')

    def get_pop_grid_geo(self, resolution: int = 100) -> tuple:
        _, _, x_origin, _, _, y_origin, *_ = self.geo_transform
        pop_transform = (x_origin, resolution, 0.0, y_origin, 0.0, -resolution)
        pop_transform = Affine.from_gdal(*pop_transform)
        return pop_transform, self.crs

    def get_pop_grid(self) -> np.ndarray:
        site_samples = [s for s in self.samples if s['site'] == 'kigali']
        m = max([s['i'] for s in site_samples]) + 1
        n = max([s['j'] for s in site_samples]) + 1
        arr = np.full((m, n, 2), fill_value=np.nan, dtype=np.float32)
        return arr

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
            else:
                self.samples = [s for s, r in zip(self.labeled_samples, random_numbers) if r >= 0.8]
        else:
            pass

        self.images = {site: self._get_planet_img(site) for site in self.sites}

        # handling transformations of data
        self.no_augmentations = no_augmentations
        self.transform = augmentations.compose_transformations(cfg.AUGMENTATION, no_augmentations)

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)
        self.length = len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        i_img, j_img, site = s['i_img'], s['j_img'], s['site']

        patch = self.images[site][i_img:i_img + 100, j_img:j_img + 100, ]
        x = self.transform(patch)

        y = s['pop']

        item = {
            'x': x,
            'y': torch.tensor([y]),
            'i': i_img,
            'j': j_img,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'


# dataset for urban extraction with building footprints
class PopInferenceDataset(AbstractPopDataset):

    def __init__(self, cfg: experiment_manager.CfgNode, year: int, nonans: bool = False):
        super().__init__(cfg)

        # handling transformations of data
        self.no_augmentations = True
        self.transform = augmentations.compose_transformations(cfg.AUGMENTATION, self.no_augmentations)

        self.samples = []
        for unit_nr in self.metadata['samples'].keys():
            if int(unit_nr) == 0 and nonans:
                continue
            self.samples.extend(self.metadata['samples'][unit_nr])

        manager = multiprocessing.Manager()
        self.samples = manager.list(self.samples)

        self.year = year
        self.img = self._get_s2_img(self.year, self.season)

        self.length = len(self.samples)

    def __getitem__(self, index):

        s = self.samples[index]
        i, j, unit, isnan = s['i'], s['j'], s['unit'], bool(s['isnan'])

        i_start, i_end = i * 10, (i + 1) * 10
        j_start, j_end = j * 10, (j + 1) * 10
        patch = self.img[i_start:i_end, j_start:j_end, ]
        x = self.transform(patch)

        y = s[f'pop{self.year}'] if f'pop{self.year}' in s.keys() else np.nan

        item = {
            'x': x,
            'y': np.nan if isnan else y,
            'unit': unit,
            'i': i,
            'j': j,
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'
