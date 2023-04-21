from typing import Dict

import torch
from torchmetrics import functional as FM


def regression_metrics(preds: torch.Tensor,
                       target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    get_classification_metrics
    Return some metrics evaluation the classification task

    Parameters
    ----------
    preds : torch.Tensor
        logits, probs
    target : torch.Tensor
        targets label

    Returns
    -------
    Dict[str, torch.Tensor]
        _description_
    """
    mse: torch.Tensor = FM.mean_squared_error(preds=preds, target=target)
    mape: torch.Tensor = FM.mean_absolute_percentage_error(preds=preds,
                                                           target=target)
    return dict(mse=mse, mape=mape)
