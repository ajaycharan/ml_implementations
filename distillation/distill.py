import torch
import torch.nn.functional
from torch import nn
# define CIFAR10Configs in experiments/cifar
# define SmallModel in this dir
# define LargeModel in this dir
# in train_valid define BatchIndex
# define tracker in root
# define experiments in root

class Configs(CIFAR10COnfigs):
    """
    Extends th CIFAR10Configs, which defines dataset related 
    configs, optimizer, and a train loop
    """
    # small model
    small: SmallModel
    # large model
    large: LargeModel
    # loss for soft targets
    soft_loss = nn.KLDivLoss(log_target=True)
    # loss for true label loss
    true_loss = nn.CrossEntropyLoss()

    temperature: float = 5.
    # weight for soft target loss
    # gradients produced by the soft tragets
    # get scaled by a factor
    # to compensate, we scale the soft targets loss by other factor
    soft_loss_wt: float = 100.
    # weight for true label loss
    true_loss_wt: float = 0.5

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        train/val step
        customized here to include distillation
        """
        # train/eval mode for small model
        self.small.train(self.mode.is_train)
        # large model in eval mode
        self.large.eval()

        # move data to device
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        # update global step (num samples processed) when train mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        # get output logits 
        # from large model
        with torch.no_grad():
            large_model_logits = self.large(data)
        # from small model
        small_model_logits = self.small(data)

        # get targets
        soft_targets = nn.functional.log_softmax(large_model_logits/self.temperature, dim=-1)
        soft_prob = nn.functional.log_softmax(small_model_logits/self.temperature, dim=-1)

        # calc loss
        soft_target_loss = self.soft_loss(soft_prob, soft_targets)
        true_target_loss = self.true_loss(small_model_logits, target)

        loss = self.soft_loss_wt * soft_target_loss + self.true_loss_wt * true_target_loss
        # log the losses
        tracker.add("soft_loss": soft_target_loss,
                    "true_loss": true_target_loss,
                    "loss": loss)
        # calc and log accuracy
        self.accuracy(small_model_logits, target)
        self.accuracy.track()

        # train model
        if self.small.is_train:
            # cacl grad
            loss.backward()
            # take optimizer step
            self.optimizer.step()
            # log model params and grads on last btach of each epoch
            if batch_idx.is_last:
                tracker.add('small_model', self.small)
            # clear grads
            self.optimizer.zero_grad()
        # save tracked metrics
        tracker.save()
    
@option(Configs.large)
def _large_model(c: Configs):
    """
    create large model
    """
    return LargeModel().to(c.device)

@option(Configs.small)
def _small_model(c: Configs):
    """
    create small model
    """
    return SmallModel().to(c.device)

def get_saved_model(run_uuid: str, checkpoint: int):
    """
    Load trained large model
    """
    from distillation.large import Configs as LargeConfigs
    # in eval mode (no recordin)
    experiment.evaluate()
    # init cfgs of large model taining
    conf = LargeConfigs()
    # load saved cfg
    experiment.configs(comf, experiment.load_configs(run_uuid))
    # set models for saving/loadin
    experiment.add_pytorch_models({'model': conf.small})
    # set the run and chekpt to load
    experiment.load(run_uuid, checkpoint)
    # start: load model, prep stuff
    experiment.start()

    return conf.model

def main(run_uuid: str, checkpoint: int):
    """
    train a small model with distillation
    """
    # load saved model -> create experiment -> create conf -> set loaded large
    # load conf -> set model -> start experiment -> run training
    large_model = get_saved_model(run_uuid, checkpoint)
    experiment.create(name='distillation', comment='cifar10')

    conf = Configs()
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam,
        'optimizer.learning_rate': 2.5e-4,
        'model': '_small_model',
    })
    experiment.add_pytorch_models({'model': conf.small})
    experiment.load(None, None)
    with experiment.start():
        conf.run()





