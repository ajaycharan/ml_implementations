# Distilling knowledge in NN
pytorch implementation of paper [distilling knowledge in a NN](https://arxiv.org/abs/1503.02531)

# overview
- large model with regularization or an ensemble of models (using dropout) generalizes better than smalm model, when trained directly on data and labels.
- use it to train small model to generalize better. since small model is faster, less compute, less memory
- output probabilities of large model provide more info than labels since other classes do have a non zero probability. eg if digit is 7, then 2 will have small prob and other digits will have even smaller. 
- use this info to train a small model beteer


