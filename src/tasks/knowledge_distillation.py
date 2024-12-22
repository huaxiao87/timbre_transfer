import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from typing import Any, Optional

import torch
import torch.nn as nn
#import pytorch_lightning as pl
from lightning.pytorch import LightningModule

import hydra

from typing import List

from src.models.ddsp_decoder import DDSP_Decoder
from src.models.decoder.hpn_decoder import HpNdecoder
from src.models.decoder.ddx7_decoder import FMdecoder
from src.models.synths.wavetable_synth import WTpNsynth
from src.models.synths.hpn_synth import HpNsynth
from src.models.synths.ddx7_synth import FMsynth
from src.distillation.distillation_methods import DistillMethod
from src.utils import utils

import inspect

def rename_keys(mydict):
         return dict((k.removeprefix('model.'), rename_keys(v) if hasattr(v,'keys') else v) for k,v in mydict.items())
    
class DistillationModule(LightningModule):
    """This LightningModule can be used to train a DDSP model with or without knowledge distillation.
    For Knowledge Distillation, the model must have a teacher and a student.
    The teacher is a pre-trained model and the student is the model to be trained.
    
    Args:
        - model: Model to be trained
        - rec_loss: reconstruction Loss function
        - learning_rate: Learning rate
        - kd_config: list of distillation method used for training the student model: 
            ['audio'] for trainig with audio distillation and ['params'] for training with parameters distillation
        - w_rec: Weight for reconstruction loss. Default 1.0 if no distillation is used
        - weights_trainable: If True, w_rec and all the distillation weights are trainable. Default False
        - **kwargs: additional parameters for knowledge distillation:
            - teacher_cfg: configuration of the pre-trained model

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: nn.Module,
        rec_loss: nn.Module,
        kd_methods: List[DistillMethod],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        w_rec = 1.0,
        weights_trainable = False,
        **kwargs
    ):
        super().__init__()  
        
        self.model = model
        self.kd_methods = kd_methods
        self.rec_loss = rec_loss
        # self.w_rec = nn.Parameter(torch.tensor(w_rec), requires_grad=weights_trainable)
        self.w_rec = w_rec
        
        self.scheduler_steps = 10000
        
        # If knowledge distillation is enabled (kd_config is not empty):
        # 1) Load teacher model and set forward method
        # 2) Overwrite model_step() method
        if self.kd_methods:
            self.load_teacher(**kwargs)
            self.forward = self._forward_kd
            self.model_step = self._model_step_kd
            
        # We must initialize distillation methods here because we can't pickle them    
        # AttributeError: Can't pickle local object 'DistillFeatures._extract_teacher_feature.<locals>.hook'
        for methods in self.kd_methods:
            methods.init_distillation(self.teacher, self.model)

        # Losses 
        self.train_loss = 0
        self.val_loss = 0
        self.test_loss = 0

        # Save hyperparameters
        self.save_hyperparameters(ignore=("model", "rec_loss")) 
        
        
    def load_teacher(self, **kwargs):
        '''Load pre-trained model'''
        self.teacher = kwargs['teacher']      
        state_dict = torch.load(kwargs['teacher_path'])
        self.teacher.load_state_dict(rename_keys(state_dict['state_dict']), strict=False)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False 
            
        if isinstance(self.model.synth, WTpNsynth) and isinstance(self.teacher.synth, WTpNsynth):
            if kwargs['copy_wavetables']:
                # Copy wavetables from teacher to student
                self.model.synth.wavetables.copy_wavetables(self.teacher.synth.wavetables.wavetables, requires_grad=kwargs['train_wavetables'])
                
        if kwargs['copy_reverb']:
            for key in self.teacher.synth.reverb.state_dict().keys():
                self.model.synth.reverb.state_dict()[key].copy_(self.teacher.synth.reverb.state_dict()[key])
                
            for param in self.model.synth.reverb.parameters():
                    param.requires_grad = kwargs['train_reverb']


    def get_sr(self):
        '''Get sample rate used by synthesizer'''
        self.model.get_sr()


    def forward(self, x: torch.Tensor):
        return self.model(x), None

    
    def _forward_kd(self, x: torch.Tensor):
        with torch.no_grad():
            t_hat = self.teacher(x)      
        x_hat = self.model(x)
        return x_hat, t_hat
       
        
    def model_step(self, batch: Any):
        x = batch
        # Compute forward pass
        x_hat = self.model(x)
        # Compute reconstruction loss
        rec_loss = self.rec_loss(x["audio"], x_hat["synth_audio"])
        return self.w_rec*rec_loss, rec_loss, {}
    
    def _model_step_kd(self, batch: Any):
        x = batch
        total_loss = 0.
        distillation_loss = {}
        # Forward pass (Student and Teacher)
        self.teacher.eval()
        with torch.no_grad():
            t_hat = self.teacher(x)
        self.model.train()
        x_hat = self.model(x)
        
        # Compute reconstruction loss
        rec_loss = self.rec_loss(x["audio"], x_hat["synth_audio"])
        
        # Compute distillation loss
        for method in self.kd_methods:
            loss, dist_loss = method(t_hat, x_hat)
            total_loss += loss
            distillation_loss.update(dist_loss)
            
        # Combine losses    
        total_loss += self.w_rec * rec_loss
        return total_loss, rec_loss, distillation_loss
    
    # def on_fit_start(self) -> None:
    #     for methods in self.kd_methods:
    #         methods.init_distillation(self.teacher, self.model)
    #     return super().on_fit_start()
            
        
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss = 0
        # Reinitalize distillation methods to ensure that everything is on the same device
        for methods in self.kd_methods:
            methods.init_distillation(self.teacher, self.model)
        

    def training_step(self, batch: Any, batch_idx: int):
        total_loss, rec_loss, distillation_loss = self.model_step(batch)

        # update and log metrics
        self.train_loss = total_loss
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/rec_loss", rec_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.kd_methods:
            for key, value in distillation_loss.items():
                self.log(f"train/kd_{key}_loss", value, on_step=False, on_epoch=True, prog_bar=True)
                
        # return loss or backpropagation will fail
        return total_loss

    def on_train_epoch_end(self):
        pass


    def validation_step(self, batch: Any, batch_idx: int):
        total_loss, rec_loss, distillation_loss = self.model_step(batch)

        # update and log metrics
        self.val_loss = total_loss
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rec_loss", rec_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.kd_methods:
            for key, value in distillation_loss.items():
                self.log(f"val/kd_{key}_loss", value, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        #acc = self.val_acc.compute()  # get current val acc
        #self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        #self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        pass

    def test_step(self, batch: Any, batch_idx: int):
        total_loss, rec_loss, distillation_loss = self.model_step(batch)

        # update and log metrics
        self.test_loss = total_loss
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/rec_loss", rec_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.kd_methods:
            for key, value in distillation_loss.items():
                self.log(f"test/kd_{key}_loss", value, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        
        optimizer = self.hparams.optimizer(params=self.model.parameters())
           
        # Create a lr scheduler 
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/loss",
                        "interval": "step",
                        "frequency": 1,
                        "strict": True,
                    },
                }
            
        return {"optimizer": optimizer}
    
  
    def lr_scheduler_step(self, scheduler, metric):
        '''There is a bug while using scheduler.interval = 'step'.
        If we set scheduler frequency > dataset.epoch_step (for example: 5 for flute dataset) 
        inside scheduler configuration, it doesn't work. 
        We must override lr_scheduler_step()'''
        if self.global_step == 200000:
                self.w_rec = 1.0
                for method in self.kd_methods:
                    method.w = 0.
        if self.global_step % self.scheduler_steps == 0:
            # self.alfa = self.weights_update*self.alfa
            # self.beta = 1-self.alfa
            # print(f"Distillation weights: alfa={self.alfa}  beta={self.beta}")
            return super().lr_scheduler_step(scheduler, metric)


    
@hydra.main(version_base=None, config_path=".", config_name="test_cfg")
def main(cfg):
    
    model = hydra.utils.instantiate(cfg.model)    
    rec_loss = hydra.utils.instantiate(cfg.rec_loss) 
    #optimizer = hydra.utils.instantiate(cfg.optimizer) 
    #scheduler = hydra.utils.instantiate(cfg.scheduler) 
    kd_config = utils.get_distillation_methods(cfg.distillation.kd_config)
        
    module = DistillationModule(model=model,
                                rec_loss=rec_loss,
                                optimizer=cfg.optimizer,
                                scheduler=cfg.scheduler,
                                kd_config=kd_config,
                                w_rec=cfg.distillation.w_rec,
                                w_audio=cfg.distillation.w_audio,
                                w_params=cfg.distillation.w_params,
                                teacher_cfg=cfg.distillation.teacher_cfg)

    #print(module.hparams)
    source_forward = inspect.getsource(module.forward)
    source_model_step = inspect.getsource(module.model_step)   
    print("FORWARD: \n",source_forward)
    print("MODEL STEP: \n", source_model_step)
    

if __name__ == "__main__":
    main()