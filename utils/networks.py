import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from utils.experiment_manager import CfgNode


def save_checkpoint(network, optimizer, cfg: CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}.pt'
    checkpoint = {
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(cfg: CfgNode, device: torch.device):
    net =  PopulationNet(cfg.MODEL)
    net.to(device)

    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}.pt'
    checkpoint = torch.load(save_file, map_location=device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    net.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return net


def load_weights(output_path: Path, config_name: str, device: torch.device):
    save_file = Path(output_path) / 'networks' / f'{config_name}.pt'
    checkpoint = torch.load(save_file, map_location=device)
    return checkpoint['network']


class PopulationNet(nn.Module):

    def __init__(self, model_cfg, enable_fc: bool = True):
        super(PopulationNet, self).__init__()
        self.model_cfg = model_cfg
        self.enable_fc = enable_fc
        pt = model_cfg.PRETRAINED
        assert (model_cfg.TYPE == 'resnet')
        if model_cfg.SIZE == 18:
            self.model = torchvision.models.resnet18(pretrained=pt)
        elif model_cfg.SIZE == 50:
            self.model = torchvision.models.resnet50(pretrained=pt)
        else:
            raise Exception(f'Unkown resnet size ({model_cfg.SIZE}).')

        new_in_channels = model_cfg.IN_CHANNELS

        if new_in_channels != 3:
            # only implemented for resnet
            assert (model_cfg.TYPE == 'resnet')

            first_layer = self.model.conv1
            # Creating new Conv2d layer
            new_first_layer = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=first_layer.out_channels,
                kernel_size=first_layer.kernel_size,
                stride=first_layer.stride,
                padding=first_layer.padding,
                bias=first_layer.bias
            )
            # he initialization
            nn.init.kaiming_uniform_(new_first_layer.weight.data, mode='fan_in', nonlinearity='relu')
            if new_in_channels > 3:
                # replace weights of first 3 channels with resnet rgb ones
                first_layer_weights = first_layer.weight.data.clone()
                new_first_layer.weight.data[:, :first_layer.in_channels, :, :] = first_layer_weights
            # if it is less than 3 channels we use he initialization (no pretraining)

            # replacing first layer
            self.model.conv1 = new_first_layer
            # https://discuss.pytorch.org/t/how-to-change-no-of-input-channels-to-a-pretrained-model/19379/2
            # https://discuss.pytorch.org/t/how-to-modify-the-input-channels-of-a-resnet-model/2623/10

        # replacing fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, model_cfg.OUT_CHANNELS)
        self.relu = torch.nn.ReLU()
        self.encoder = torch.nn.Sequential(*(list(self.model.children())[:-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_fc:
            x = self.model(x)
            x = self.relu(x)
        else:
            x = self.encoder(x)
            x = x.squeeze()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 5, 224, 224)
    model = torchvision.models.vgg16(pretrained=False)  # pretrained=False just for debug reasons
    first_conv_layer = [nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    first_conv_layer.extend(list(model.features))
    model.features = nn.Sequential(*first_conv_layer)
    output = model(x)