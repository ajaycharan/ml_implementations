# Distilling knowledge in NN
pytorch implementation of paper [distilling knowledge in a NN](https://arxiv.org/abs/1503.02531)

# overview
- large model with regularization or an ensemble of models (using dropout) generalizes better than smalm model, when trained directly on data and labels.
- use it to train small model to generalize better. since small model is faster, less compute, less memory
- output probabilities of large model provide more info than labels since other classes do have a non zero probability. eg if digit is 7, then 2 will have small prob and other digits will have even smaller. 
- use this info to train a small model beteer

## soft targets
The large model's output prob distri (soft targets)

## training
Train the small model to minimize the cross entropy or KL div between its output prob distri and soft targets. The prob is computed as softmax:
$$ q_i = \frac{\exp(z_i)}{\sum_{j} \exp(z_j)} $$ 
### problem
prob assigned by large model to other classes are too small and don't contrib to the loss. So apply a temp `T` to "soften" the probs:
$$ q_i = \frac{\exp(z_i/T)}{\sum_{j} \exp(z_j/T)} $$

### second loss term
Add this to predict the actual labels when train small model. The loss then is wtd sum of soft targets and actual tragets.

## transfer set
dataset for distillation. Suggested to use the train data here.
