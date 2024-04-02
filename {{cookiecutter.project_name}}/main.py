from argparse import ArgumentParser

import yaml
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import MNIST

from components.callbacks import get_callbacks
from components.config import CallBacksConfig, ModelConfig, OptimizerConfig
from models.model_lit import MNISTModel


def train():
    parsers = ArgumentParser()
    parsers.add_argument("--config-path", dest="config_path")
    parsers.add_argument("--accelerator", dest="accelerator", default="cpu")
    parsers.add_argument("--device", dest="device", default="auto")
    parsers.add_argument("--max-epoch", dest="max_epoch", default=50)

    args = parsers.parse_args()
    with open(file=args.config_path, mode="r", encoding="utf8") as f:
        config = yaml.load(stream=f, Loader=yaml.FullLoader)

    callbacks_config = CallBacksConfig(**config["callbacks"])
    model_config = ModelConfig(**config["model_config"])
    optimizer_config = OptimizerConfig(**config["optimizer_config"])
    callbacks = get_callbacks(config=callbacks_config)

    trainset = MNIST(
        root="data/MNIST", train=True, transform=T.ToTensor(), download=False
    )
    testset = MNIST(
        root="data/MNIST", train=False, transform=T.ToTensor(), download=False
    )

    trainloader = DataLoader(dataset=trainset, batch_size=32)
    testloader = DataLoader(dataset=testset, batch_size=32)

    model = MNISTModel(model_config=model_config, optimizer_config=optimizer_config)
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.device,
        callbacks=callbacks,
        max_epochs=int(args.max_epoch),
    )

    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader)


if __name__ == "__main__":
    train()
