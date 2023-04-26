# Define callbacks here
from pytorch_lightning.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor="loss", min_delta=0, patience=3)
