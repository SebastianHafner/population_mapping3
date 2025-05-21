import torch
from pathlib import Path
from utils import networks, datasets, parsers, experiment_manager, geofiles
from utils.experiment_manager import CfgNode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inference(cfg: CfgNode):
    net = networks.load_checkpoint(cfg, device)
    net.eval()

    run_type = 'test'
    dataset = datasets.SurveyDataset(cfg=cfg, run_type=run_type, no_augmentations=True)
    for index in range(len(dataset)):
        item = dataset.__getitem__(index)
        site, grid_id = item['site'], item['id']
        if site == 'nairobi' and grid_id == 22:
            debug = True
        img = item['x'].to(device)
        if torch.sum(img).item() == 0:
            print(index)

        with torch.no_grad():
            pred = net(img.unsqueeze(0))

        pred = pred.detach().cpu().squeeze().item()
        label = item['y'].cpu().item()
        print(pred, label)



if __name__ == '__main__':
    args = parsers.inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    inference(cfg)
