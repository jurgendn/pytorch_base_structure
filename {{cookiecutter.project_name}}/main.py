from argparse import ArgumentParser

import yaml
from components.callbacks import get_callbacks
from components.config import (
    CallBacksConfig,
    LoggerConfig,
    ModelConfig,
    OptimizerConfig,
)
from models.model_lit import MNISTModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import MNIST


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

    model = MNISTModel(model_config=model_config, optimizer_config=optimizer_config)
    print(model)
    trainer_logger = WandbLogger(
        save_dir=logger_config.save_dir,
        project=logger_config.project,
    )
    print(trainer_logger)
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.device,
        logger=trainer_logger,
        callbacks=callbacks,
        max_epochs=int(args.max_epoch),
        log_every_n_steps=1,
    )

    trainer.fit(model=model, train_dataloaders=testloader, val_dataloaders=testloader)


if __name__ == "__main__":
    train()
