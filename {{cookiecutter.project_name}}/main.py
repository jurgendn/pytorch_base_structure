from argparse import ArgumentParser

import yaml
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import MNIST

from components.callbacks import get_callbacks
from components.config import (
    CallBacksConfig,
    LoggerConfig,
    ModelConfig,
    OptimizerConfig,
)
from components.loggers import WandBLogger
from components.trainer import Trainer
from models.model import MNISTModel


def train():
    parsers = ArgumentParser()
    parsers.add_argument("--config-path", dest="config_path")
    parsers.add_argument("--accelerator", dest="accelerator", default="cpu")
    parsers.add_argument("--device", dest="device", default="auto")
    parsers.add_argument("--max-epoch", dest="max_epoch", default=50)
    parsers.add_argument("--download", dest="download", default=True)

    args = parsers.parse_args()
    with open(file=args.config_path, mode="r", encoding="utf8") as f:
        config = yaml.load(stream=f, Loader=yaml.FullLoader)

    callbacks_config = CallBacksConfig(**config["callbacks"])
    model_config = ModelConfig(**config["model_config"])
    optimizer_config = OptimizerConfig(**config["optimizer_config"])
    logger_config = LoggerConfig(**config["logger_config"])
    callbacks = get_callbacks(config=callbacks_config)
    trainset = MNIST(
        root="data/MNIST", train=True, transform=T.ToTensor(), download=args.download
    )
    testset = MNIST(
        root="data/MNIST", train=False, transform=T.ToTensor(), download=args.download
    )

    trainloader = DataLoader(dataset=trainset, batch_size=32)
    testloader = DataLoader(dataset=testset, batch_size=32)

    model = MNISTModel(model_config=model_config)
    trainer_logger = WandBLogger(project_name=logger_config.project)
    trainer = Trainer(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        loss_fn=F.cross_entropy,
        logger=trainer_logger,
        callbacks=callbacks,
    )

    trainer.train(train_loader=testloader, val_loader=testloader, epochs=10)


if __name__ == "__main__":
    train()
