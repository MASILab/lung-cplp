import os, argparse
import torch
from torch.utils.data import DataLoader
from lungcplp.datasets import PHASE, DataModule
from lungcplp.config import Config
import lungcplp.models.models as MODELS, lungcplp.train as TRAINER
from lungcplp.embeddings import save_class_embeddings
from lungcplp.utils import find_best_checkpoint, auc_from_df, bootsrap, read_tabular, seed_everything, prepare_device
import lungcplp.definitions as D

def init_model(model_name, config):
    if model_name == "clip_base":
        model = MODELS.__dict__[model_name](
            embed_dim=config.embed_dim or 64,
            vision_width=config.img_dim,
            vocab_size=config.expr_dim,
            transformer_width=config.transformer_width,
        )
    else:
        model = MODELS.__dict__[model_name](
            embed_dim=config.embed_dim,
            vision_width=config.img_dim,
            vocab_size=config.expr_dim,
            transformer_width=config.transformer_width,
            classifier_depth=config.classifier_depth,
        )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("phase", choices=PHASE.__annotations__.keys()) # train, pretrain, cv, test, predict
    parser.add_argument("--cohort", default=None, type=str)
    parser.add_argument("--silent", default=False, action='store_true')
    parser.add_argument("--local_drive", default=False, action='store_true')
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--predictions", default=None, type=str)
    parser.add_argument("--subset", default=1.0, type=float)
    args = parser.parse_args()
    
    seed_everything(D.RANDOM_SEED)
    torch.set_float32_matmul_precision('high')
    
    config = Config(args.config)
    device, _ = prepare_device(config.n_gpu)
    dataroot = args.local_drive or "./data"
    model = init_model(config.model_name, config)
    if args.cohort is not None:
        config.cohort = args.cohort
    
    if args.phase in [PHASE.train, PHASE.pretrain]:
        datamodule = DataModule(config, args.phase, dataroot=dataroot)
        train_dataloader = datamodule.train_dataloader()
        trainer = TRAINER.__dict__[config.trainer_name](model, config, args.phase, device, train_dataloader, silent=args.silent)
        trainer.train()
        
    if args.phase == PHASE.test:
        datamodule = DataModule(config, args.phase, dataroot=dataroot)
        test_dataloader = datamodule.test_dataloader()
        trainer = TRAINER.__dict__[config.trainer_name](model, config, args.phase, device, test_dataloader, silent=True)
        
        if args.checkpoint is None:
            checkpoint_path = find_best_checkpoint(os.path.join(D.RUNS_DIR, config.id, "ckpts"), metric=config.monitor)
        else:
            checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Predicting from checkpoint: {checkpoint_path}")
        trainer.test()
        
    if args.phase == PHASE.predict:
        datamodule = DataModule(config, args.phase, dataroot=dataroot, subset=args.subset)
        predict_dataloader = datamodule.predict_dataloader()
        trainer = TRAINER.__dict__[config.trainer_name](model, config, args.phase, device, predict_dataloader, silent=True)

        if args.checkpoint is None:
            checkpoint_path = find_best_checkpoint(os.path.join(D.RUNS_DIR, config.id, "ckpts"), metric=config.monitor)
        else:
            checkpoint_path = args.checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Predicting from checkpoint: {args.checkpoint}")
        trainer.predict()
    
    if args.phase == PHASE.zeroshot:
        if config.compute_text_embeddings:
            save_class_embeddings(config.cohort, model="stella_en_400M_v5", batch_size=config.batch_size, class_label=config.label, local_drive=args.local_drive)
        datamodule = DataModule(config, args.phase, dataroot=dataroot)
        predict_dataloader = datamodule.predict_dataloader()
        trainer = TRAINER.__dict__[config.trainer_name](model, config, args.phase, device, predict_dataloader, silent=True)
        
        print(f"Predicting from checkpoint: {config.encoder_ckpt}")
        trainer.test()

    if args.phase == "bootstrap":
        if args.predictions is None:
            fname = f"test_{args.cohort}.csv"
            pred_df = read_tabular(os.path.join(D.RUNS_DIR, config.id, "out", fname))
        else:
            pred_df = read_tabular(args.predictions)
        metrics = bootstrap(
            pred_df,
            agg=lambda x: auc_from_df(x, cols=('label', 'prob')),
        )
        print(metrics)

        
if __name__ == "__main__":
    main()