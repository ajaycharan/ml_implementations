import torch

from language_model.sampling import Sampler

class GreedySampler(Sampler):
    """
    ## Sample most likely token from the distribution of logits
    """
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)
