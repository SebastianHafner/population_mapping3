import torch
from pathlib import Path
from utils import networks, datasets, parsers, experiment_manager, geofiles
from utils.experiment_manager import CfgNode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(cfg: CfgNode):
    net = networks.load_checkpoint(cfg, device)
    net.eval()

    data = []

    for run_type in ['train', 'test', 'unlabeled']:
        dataset = datasets.SurveyDataset(cfg=cfg, run_type=run_type)
        for index in range(len(dataset)):
            item = dataset.__getitem__(index)
            img = item['x'].to(device)

            with torch.no_grad():
                pred = net(img.unsqueeze(0))

            pred = pred.detach().cpu().squeeze().item()
            label = item['y'].item()
            site, grid_id = item['site'], item['id']
            data.append({
                'model': pred,
                'label': label,
                'site': site,
                'id': grid_id,
                'split': run_type,
            })

    out_file = Path(cfg.PATHS.OUTPUT) / 'inference' / f'inference_{cfg.NAME}.json'
    geofiles.write_json(out_file, data)


if __name__ == '__main__':
    args = parsers.inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    inference(cfg)
