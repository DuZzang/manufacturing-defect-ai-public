"""STFPM (Student-Teacher Feature Pyramid Matching) model builder using Anomalib API."""

from typing import Any, List, Optional, Sequence, Union

import torch
from anomalib.models import Stfpm


DEFAULT_STFPM_LAYERS: Sequence[str] = ("layer1", "layer2", "layer3")


def build_stfpm_model(
    backbone: str = "resnet18",
    layers: Optional[Sequence[str]] = ("layer1", "layer2", "layer3"),
    learning_rate: Optional[float] = None,
    momentum: Optional[float] = None,
    weight_decay: Optional[float] = None,
    pre_processor: Union[bool, Any] = False,
    post_processor: Union[bool, Any] = False,
    **kwargs: Any,
) -> Stfpm:
    """Builds and initializes an STFPM model using Anomalib API.

    Args:
        backbone: Feature extractor backbone architecture (e.g., 'resnet18' or 'resnet34').
            Default is 'resnet18'.
        layers: Layers used for multi-scale feature pyramid matching.
            Default is ('layer1', 'layer2', 'layer3').
        learning_rate: Optional learning rate for Student model SGD optimizer.
        momentum: Optional momentum for Student model SGD optimizer.
        weight_decay: Optional weight decay for Student model SGD optimizer.
        pre_processor: Pre-processor module or bool flag. Default is False
            (delegating preprocessing to dedicated Printer3DPreprocessor).
        post_processor: Post-processor module or bool flag. Default is False
            (preserving raw distance scores).
        **kwargs: Additional keyword arguments passed directly to Stfpm.

    Returns:
        Configured Stfpm model instance.
    """
    if layers is None:
        selected_layers: Sequence[str] = DEFAULT_STFPM_LAYERS
    else:
        selected_layers = list(layers)

    model = Stfpm(
        backbone=backbone,
        layers=selected_layers,
        pre_processor=pre_processor,
        post_processor=post_processor,
        **kwargs,
    )

    if any(p is not None for p in (learning_rate, momentum, weight_decay)):
        lr = float(learning_rate) if learning_rate is not None else 0.4
        mom = float(momentum) if momentum is not None else 0.9
        wd = float(weight_decay) if weight_decay is not None else 0.001

        def configure_custom_optimizers():
            return torch.optim.SGD(
                params=model.model.student_model.parameters(),
                lr=lr,
                momentum=mom,
                dampening=0.0,
                weight_decay=wd,
            )

        model.configure_optimizers = configure_custom_optimizers
        model.learning_rate = lr
        model.momentum = mom
        model.weight_decay = wd

    return model

