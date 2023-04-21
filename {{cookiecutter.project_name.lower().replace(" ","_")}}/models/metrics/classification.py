from typing import Dict

import torch
from torchmetrics import functional as FM


def classification_metrics(
        preds: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        average: str = 'macro',
        task: str = 'multiclass') -> Dict[str, torch.Tensor]:
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
    f1 = FM.f1_score(preds=preds,
                     target=target,
                     num_classes=num_classes,
                     task=task,
                     average=average)
    recall = FM.recall(preds=preds,
                       target=target,
                       num_classes=num_classes,
                       task=task,
                       average=average)
    precision = FM.precision(preds=preds,
                             target=target,
                             num_classes=num_classes,
                             task=task,
                             average=average)
    return dict(f1=f1, precision=precision, recall=recall)
