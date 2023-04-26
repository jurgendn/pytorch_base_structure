# PyTorch Base Structure Generator

## System Requirements

I built all models on a docker environment to ensure the training process is consistent on all systems. Please install these packages before building any images.

- Docker
- docker-compose

If CUDA is available, we can do the training process is Docker environment, with these additional packages:

- docker-compose >= 2.8
- nvidia-container-toolkit
- CUDA

## Project Structure

I use `PyTorch` as the main deep learning framework and `PyTorch Lightning` to write code faster and shorter. This structure is built to use with `PyTorch` and `PyTorch Lightning`, and may not be appropriate for other frameworks.

Some main prerequisite packages are used in this template:

```pip-requirements
dynaconf==3.1.12
pytorch-lightning
```

This folder tree represents the structure of my projects.

```bash
.
├── data
├── components
│   ├── callbacks.py
│   └── data_module.py
├── config
│   └── config.yaml
├── models
│   ├── __init__.py
│   ├── base_model
│   │   ├── classification.py
│   │   ├── gan.py
│   │   └── regression.py
│   ├── metrics
│   │   ├── classification.py
│   │   └── regression.py
│   ├── model_lit.py
│   └── modules
│       └── sample_torch_module.py
├── tests
│   └── test_resource.py
└── utils
├── config.py
├── main.py
├── requirements.txt
├── docker-compose.yml
├── startup.sh
├── README.md
├── AUTHORS.rst
```

There are some main components in this architecture.

There are some main components in this architecture.

### Config

We use [`dynaconf`](https://www.dynaconf.com/) to parse and load configuration files. `Dynaconf` supports many file types, such as `yaml, json, toml, ini, py`, and `env`. For more detail, please read the documentation.

The `./config/` folder contains all the config files, and `dynaconf` will load configuration from these files in `./config.py`

```python
from dynaconf import Dynaconf

CFG = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['./config/config.yaml'],
)
```

The `CFG` variable now contains all the configurations that are declared in `./config/config.yaml`

### Models

This module contains the definitions of the models used in this project. Note that we can divide our models as small as we want, but there are some pre-defined modules written by `PyTorch Lightning` in `./models/base_model/` that we can use to build models faster. Small components should be defined in `./models/modules/`, for example, some custom CNN, loss functions,...

```bash
├── models
│   ├── __init__.py
│   ├── base_model
│   │   ├── classification.py
│   │   ├── gan.py
│   │   └── regression.py
│   ├── metrics
│   │   ├── classification.py
│   │   └── regression.py
│   ├── model_lit.py
│   └── modules
│       └── sample_torch_module.py
```

In the `base_model`, I defined three modules for 3 different tasks: _classification, regression_, and _GAN_. In each model, 3 abstract methods need to overwrite when using:

- The constructor: `__init__`
- The forwarding method: `forward`
- The training step: `training_step`

Of course, we can overwrite any methods in this model, however, to make it run, we simply overwrite the 3 above methods.

### Components

This folder contains everything necessary to train a model: _callbacks, datamodule_

The `callbacks` is optional and the `data_module` is included in the `Dataset` class and `LightningDataModule` class. Thank `PyTorch Lightning`, we can define both DataLoader and sampling methods in one class.

For more information, please read the [documentation](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html).

### Logger

The `logging` that is used in this project is `NeptuneLogger`. This is a personal preference, and we can choose different logging providers among \*\*Neptune.ai, CometML, WanDB, and MLFlow,...

The `logger` instance is defined directly in `main.py`. The remote logger will need an `API key` and project information, and this information should be declared in `env`

### Containerization

We run the training job on the Docker environment. To reduce the build time, I created my image. This custom image contains some basic packages:

- PyTorch and related modules
- opencv-headless
- CUDA

The image is placed on [Dockerhub](https://hub.docker.com/r/gr000a1/torch-gpu/tags)

```bash
.
|-- .docker
|   |-- Dockerfile
|   `-- startup.sh
|-- docker-compose.yml
|-- .env
|-- .dockerignore
```

This table gives the description of the file/folder in containerization.

| File/Folder         |                          Description |
| ------------------- | -----------------------------------: |
| ./docker            |   Includes Dockerfile and startup.sh |
| docker-compose.yaml |                  Docker Compose file |
| .dockerignore       | Define what not to copy to the image |

### Test

We have a folder `test/` here containing all test cases, to check whether all the tests are satisfied or not.

The testing framework in this base project is [`PyTest`](https://docs.pytest.org/en/7.2.x/). Run all tests in the folder by placing this line in `./docker/startup.sh`:

```bash
pytest tests/
```

### Training model

To train a model, assume we place all necessary scripts are placed in `./main.py`.

The `./.docker/startup.sh` script file should contains:

```bash
# /bin/bash

pytest tests/
python main.py
```
