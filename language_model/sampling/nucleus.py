"""
---
title: implementation of nucleus sampler from the paper
[The curious case of Neural text degeneration](https://arxiv.org/abs/1904.09751) 
---
The paper discusses the problems with other sampling methods such as Beam search,
pure sampling, temperature sampling, top-k sampling.
It introduces nucleus sampling which practically performs better than others.
It first picks a subset of the vocabulary $V^{(p)} \subset V$,
Where $V^{(p)}$ is smallest subset of tokens such that:
$$\sum_{x_i \in V^{(p)}} P(x_i | x_{1:i-1}) \ge p$$
That is, we pick the highest probable tokens until the sum of their probabilities is less than $p$
Then we sample from the selected tokens 
"""
import torch
from torch import nn

from language_model.sampling import Sampler

class NucleusSampler(Sampler):
    """
    ## Nucleus sampler
    Picks a subset vocabulary: the set of highest probable tokens until their sum is less than $p$
    Then we sample from selected tokens 
    """
    def __init__(self, p: float, sampler: Sampler):
        """
        :param p: is the sum of probabilities of the tokens to pick
        :param sampler: is the sample to use for the selected tokens
        """
        self.p = p
        self.sampler = sampler
        # Softmax to compute $P(x_i | x{1:i-1})$ from the logits
        self.softmax = nn.Softmax(dim=-1)
    
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from logits with nucleus sampling
        """
        # get probabilities $P(x_i | x_{1:i-1})$
        probs = self.softmax(logits)

        # sort the probabilities in descending order
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        # Get the cumulative sum of probabilities in the sorted order
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        # find the cumulative sum less than $p$
        nucleus = cum_sum_probs < self.p
        # prepend ones so that we add one token after the minimum number
        # of tokens with cumulative probability less than $p$
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
        # get log probabilities and mask out non-nucleus
        sorted_log_probs = torch.log(sorted_probs)
        sorted_log_probs[~nucleus] = float('-inf')
        # sample from the sampler
        sample_sorted_indices = self.sampler(sorted_log_probs)
        # get the actual indices
        res = indices.gather(-1, sample_sorted_indices.unsqueeze(-1))
        
        return res.squeeze(-1)
    