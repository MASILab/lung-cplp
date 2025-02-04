import numpy as np, math, pandas as pd
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
import os
from abc import abstractmethod
from tqdm import tqdm
import wandb
from lungcplp.datasets import PHASE
import lungcplp.utils as utils
from lungcplp.datasets import PHASE
# from lungcplp.models.models import convert_weights
from sklearn.metrics import roc_auc_score

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, config, phase, silent=False, out_suffix=None):
        self.config = config
        self.silent = silent
        self.out_suffix = out_suffix
        self.logger = config.get_logger('logging', config.verbosity)

        self.model = model
        self.scaler = torch.amp.GradScaler(enabled=True) # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        
        self.epochs = config.epochs
        self.save_period = config.save_period
        self.monitor = config.monitor

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = np.inf if self.mnt_mode == 'min' else -np.inf
            self.early_stop = np.inf if config.early_stop is None else config.early_stop
            if self.early_stop <= 0:
                self.early_stop = np.inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        if not self.silent:
            # setup visualization writer instance
            wandb.init(project='lungcplp', name=config.id, dir=config.log_dir, config={
                'batch_size': config.batch_size,
                'dataset': config.dataset,
                'cohort': config.cohort,
                'model': config.model_name,
            })

        # # resume training from checkpoint
        # if config.resume is not None:
        #     self._resume_checkpoint(config.resume)

    def wandb_log(self, metrics):
        if not self.silent:
            wandb.log(metrics)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        # with torch.autograd.detect_anomaly():
        not_improved_count = 0
        self.best_ckpt = None
        for epoch in range(self.start_epoch, self.epochs + 1):
            log = self._train_epoch(epoch)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                # type error if mnt_metric is None
                except KeyError or TypeError: 
                    # self.logger.warning("Warning: Metric '{}' is not found. "
                    #                     "Consider disabling monitoring".format(self.mnt_metric))
                    # self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best=True
                    self.best_ckpt = self._save_checkpoint(epoch, save_best=best)
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False)
                
    def test(self) -> pd.DataFrame:
        pred_df = self._test()
        suffix = self.config.cohort if self.out_suffix is None else self.out_suffix
        dst = os.path.join(self.config.out_dir, f"test_{suffix}.csv")
        pred_df.to_csv(dst, index=False)
        
        auc = roc_auc_score(pred_df['label'], pred_df['prob'])
        print(f"Test AUC: {auc}")
        
    def _save_checkpoint(self, epoch, save_best=None):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        mnt_metric_str = str(self.mnt_metric).replace('/', '_') # in case metric has a '/' like 'train/loss'
        if save_best:
            fpath = os.path.join(self.checkpoint_dir, f'model_best-epoch={epoch}-{mnt_metric_str}={self.mnt_best:.3f}.pth')
            # remove previous best checkpoint
            if self.best_ckpt is not None:
                os.remove(self.best_ckpt)
        else:
            fpath = os.path.join(self.checkpoint_dir,f'epoch={epoch}-{mnt_metric_str}={self.mnt_best:.3f}.pth')
        torch.save(state, fpath)

        return fpath

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        # if checkpoint['config']['arch'] != self.config['arch']:
        #     self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
        #                         "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        #     self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
        #                         "Optimizer parameters not being resumed.")
        # else:
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

class CLIPTrainer(BaseTrainer):
    def __init__(self, model, config, phase, device,
                data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, **kwargs):
        super().__init__(model, config, phase, **kwargs)
        self.device = device
        self.model.to(device)
        self.data_loader = data_loader
        self.steps_per_epoch =len(self.data_loader)
        self.log_every_n_steps = config.log_every_n_steps
        total_training_steps = self.steps_per_epoch * config.epochs

        # configer optimizers
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
        if config.lr_schedule is not None:
            self.lr_scheduler = utils.__dict__[config.lr_schedule](self.optimizer, self.config, total_training_steps)
        else:
            self.lr_scheduler = utils.cosine_annealing_warmup_restarts(self.optimizer, self.config, total_training_steps)

        self.loss_img = nn.CrossEntropyLoss()
        self.loss_expr = nn.CrossEntropyLoss()

        # resume training from checkpoint
        if config.resume is not None:
            self._resume_checkpoint(config.resume)
        
        self.metrics = {}

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.metrics = {"epoch": epoch} # reset metrics
        running_loss = 0.0
        with tqdm(self.data_loader, unit="batch", total=len(self.data_loader), leave=False) as batches:
            batches.set_description(f"Epoch {epoch}")
            for batch_idx, batch in enumerate(batches):
                
                img, expr = batch['data']
                img, expr = img.to(self.device), expr.to(self.device)
                times, padding = batch['times'].to(self.device), batch['padding'].to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                    logits_per_image, logits_per_text = self.model(img, expr, padding, times)
                    labels = torch.arange(len(img), dtype=torch.long, device=logits_per_image.device)

                    loss = (self.loss_img(logits_per_image, labels) + self.loss_expr(logits_per_text, labels)) / 2
                    acc = (logits_per_image.argmax(dim=1) == labels).float().mean()
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                
                # if self.device == "cpu":
                #     self.optimizer.step()
                # else:
                #     convert_models_to_fp32(self.model)
                #     self.optimizer.step()
                #     convert_weights(self.model)
                
                # logging
                running_loss += loss.item()
                total_steps = batch_idx + 1 + (self.steps_per_epoch*epoch)
                self.metrics.update({
                    "loss": loss.item(),
                    "acc": acc.item(),
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "total_steps": total_steps,
                })
                
                if total_steps % self.log_every_n_steps == 0:
                    self.wandb_log(self.metrics)
                batches.set_postfix(loss=f"{loss.item():.4f}")

            self.metrics.update({"loss": running_loss / self.steps_per_epoch}) # update loss to be avg per batch
            return self.metrics

    def predict(self):
        Path(os.path.join(self.config.out_dir, "features",)).mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for batch in self.data_loader:
                img, expr = batch['data']
                img, expr = img.to(self.device), expr.to(self.device)
                times, padding = batch['times'].to(self.device), batch['padding'].to(self.device)
                
                img_features = self.model.encode_image(img, padding, times)
                text_features = self.model.encode_text(expr)
                
                features = torch.cat([img_features, text_features], dim=1)
                
                pids = batch['pid']
                for i, pid in enumerate(pids):
                    np.save(os.path.join(self.config.out_dir, "features", f"img_text_{pid}.npy"), features[i].detach().cpu().numpy())

class FinetuneCLIPTrainer(BaseTrainer):
    def __init__(self, model, config, phase, device,
            data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, **kwargs):
        super().__init__(model, config, phase, **kwargs)
        self.device = device
        self.model.to(device)
        
        # init pretrained encoders
        if config.encoder_ckpt is not None:
            print(f"Loading pretrained encoders from :{config.encoder_ckpt}")
            self.model.load_encoders(config.encoder_ckpt)
        if (self.config.encoder_ckpt is not None) and (config.encoder_lr is None):
            self.model.freeze_encoders()
            
        self.data_loader = data_loader
        self.steps_per_epoch = len(self.data_loader)
        self.log_every_n_steps, self.val_every_n_epochs = config.log_every_n_steps, config.val_every_n_epochs

        # configer optimizers
        param_groups = self.model.parameters() if config.encoder_lr is None else self.model.set_encoder_lr_param_groups(config.encoder_lr)
        self.optimizer = optim.AdamW(param_groups, lr=self.config.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
        self.loss = nn.BCEWithLogitsLoss()
        self.one_hot = lambda x: nn.functional.one_hot(x, num_classes=2).to(torch.float32)
        if phase == PHASE.zeroshot:
            self.predict_as_onehot = True
        else:
            self.predict_as_onehot = config.predict_as_onehot

        # resume training from checkpoint
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def _train_epoch(self, epoch):
        self.metrics = {"epoch": epoch} # reset metrics
        running_loss = 0.0
        with tqdm(self.data_loader, unit="batch", total=len(self.data_loader), leave=False) as batches:
            batches.set_description(f"Epoch {epoch}")
            for batch_idx, batch in enumerate(batches):
                
                x, label = utils.unpack_batch(batch, self.config, self.device)
                # if self.config.dataset == 'Imgft_Dataset':
                #     img = batch['data'].to(self.device)
                # else:
                #     img, expr = batch['data']
                #     img, expr = img.to(self.device), expr.to(self.device)
                # times, padding, label = batch['times'].to(self.device), batch['padding'].to(self.device), batch['label'].to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                    y_hat = self.model(*x)
                    if self.predict_as_onehot:
                        y = self.one_hot(label).float()
                        acc = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
                    else:
                        y = label.float().unsqueeze(1)
                        acc = (y_hat.argmax(dim=1) == y).float().mean()
                    loss = self.loss(y_hat, y)
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                running_loss += loss.item()
                total_steps = batch_idx + 1 + (self.steps_per_epoch*epoch)
                self.metrics.update({
                    "train/loss": loss.item(),
                    "train/acc": acc.item(),
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "total_steps": total_steps,
                })
                
                if total_steps % self.log_every_n_steps == 0:
                    self.wandb_log(self.metrics)
                batches.set_postfix(loss=f"{loss.item():.4f}")
            
            # validation
            if epoch % self.val_every_n_epochs == 0:
                val_metrics = self._valid_epoch()
                self.metrics.update(val_metrics)

            self.metrics.update({"train/loss": running_loss / self.steps_per_epoch}) # update loss to be avg per batch
            return self.metrics

    def _valid_epoch(self):
        self.model.eval()
        y_hats, ys = [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                x, label = utils.unpack_batch(batch, self.config, self.device)
                # img, expr = batch['data']
                # img, expr = img.to(self.device), expr.to(self.device)
                # times, padding, label = batch['times'].to(self.device), batch['padding'].to(self.device), batch['label'].to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                    logit = self.model(*x)
                    
                if self.predict_as_onehot:
                    y_hat = torch.softmax(logit, dim=1)
                    y = self.one_hot(label).float()
                    acc = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
                else:
                    y_hat = torch.sigmoid(logit)
                    y = label.float().unsqueeze(1)
                    acc = ((y_hat > 0.5) == y).float().mean()
                y_hats.append(y_hat)
                ys.append(y)

            y_hats, ys = torch.cat(y_hats), torch.cat(ys)
            loss = self.loss(y_hats, ys)
            
            auc = auc_from_tensor(ys, y_hats)
            val_metrics = {
                "val/loss": loss.item(),
                "val/acc": acc.item(), 
                "val/auc": auc,
                "total_steps": self.metrics['total_steps'],
                "epoch": self.metrics['epoch'],
            }
            self.wandb_log(val_metrics)
            return val_metrics
    
    def _test(self):
        self.model.eval()
        test_pred = []
        with torch.no_grad():
            # Extracting regression coefficeints
            if self.config.regression:
                weights = self.model.linear.weight.detach().cpu().numpy()
                feature_ranking, importance = utils.plot_feature_importance(weights, out_dir=self.config.root_dir)
                print(f"Top 10 features: {list(zip(feature_ranking[:10], importance[:10]))}")
            
            for batch_idx, batch in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
                x, labels = utils.unpack_batch(batch, self.config, self.device)
                # img, expr = batch['data']
                # img, expr = img.to(self.device), expr.to(self.device)
                # times, padding, labels = batch['times'].to(self.device), batch['padding'].to(self.device), batch['label'].to(self.device)
                pids = batch['pid']
                
                logit = self.model(*x)
                for i, pid, in enumerate(pids):
                    label = labels[i].detach().cpu().item()
                    if self.predict_as_onehot:
                        y_hat = torch.softmax(logit, dim=1)
                        prob = y_hat[i][1].detach().cpu().item()
                    else:
                        y_hat = torch.sigmoid(logit)
                        prob = y_hat.flatten()[i].detach().cpu().item()
                    test_pred.append(TestPred(pid=pid, label=label, prob=prob))
            
            return pd.DataFrame(test_pred)
    
    def predict(self):
        Path(os.path.join(self.config.out_dir, "features",)).mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for batch in self.data_loader:
                img, expr = batch['data']
                img, expr = img.to(self.device), expr.to(self.device)
                features = torch.concat([img[:, 0], expr[:,0]], dim=-1)
                pids = batch['pid']
                for i, pid in enumerate(pids):
                    np.save(os.path.join(self.config.out_dir, "features", f"img_text_{pid}.npy"), features[i].detach().cpu().numpy())
        

def auc_from_tensor(y: torch.tensor, y_hat: torch.tensor):
    y_hat, y = y_hat.cpu().numpy(), y.cpu().numpy()
    if len(np.unique(y[:,0])) == 1:
        print("Only one class in validation set. Skipping AUC calculation.")
        return None
    else:
        return roc_auc_score(y, y_hat)

from typing import TypedDict
class TestPred(TypedDict):
    pid: str
    label: torch.Tensor
    prob: torch.Tensor