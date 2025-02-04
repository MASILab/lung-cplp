import os, argparse, yaml
import logging
import logging.config
from pathlib import Path
from lungcplp.utils import read_json
import lungcplp.definitions as D

class Config():
    def __init__(self, configf) -> None:
        config = self.load_config(configf)
        
        self.id = os.path.splitext(os.path.basename(configf))[0]
        self.root_dir = os.path.join(D.RUNS_DIR, self.id)
        self.save_dir = os.path.join(self.root_dir, 'ckpts')
        self.log_dir = os.path.join(self.root_dir, 'logs')
        self.out_dir = os.path.join(self.root_dir, 'out')
        Path(self.root_dir).mkdir(parents=True, exist_ok=True)
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
                
        self.dataset = config['data']['dataset']
        self.cohort = config['data']['cohort']
        self.img_dim = config['data']['img_dim'] # set to 0 if want to exclude this modality
        self.expr_dim = config['data']['expr_dim'] # set to 0 if want to exclude this modality
        self.date_format = config['data']['date_format']
        self.val_split = config['data']['val_split']
        
        self.log_every_n_steps = config['logging']['log_every_n_steps']
        self.val_every_n_epochs = config['logging']['val_every_n_epochs']
        self.verbosity = config['logging']['verbosity']

        self.model_name = config['model']['model_name']
        self.embed_dim = config['model']['embed_dim']
        self.slen = config['model']['slen']
        
        self.trainer_name = config['trainer']['trainer_name']        
        self.batch_size = config['trainer']['batch_size']
        self.lr = config['trainer']['lr']
        self.epochs = config['trainer']['epochs']
        self.save_period = config['trainer']['save_period']
        self.monitor = config['trainer']['monitor']
        self.resume = config['trainer']['resume']
        self.n_gpu = config['trainer']['n_gpu']

        # optional params
        optional_params = [
            ('data', 'label'),
            ('data', 'n_splits'),
            ('data', 'n_cohort'), # dataset size (including val set)
            ('data', 'n_pretrain'), # ssl dataset size
            ('data', 'date_format'),
            ('data', 'fpath_from_df'), # use the fpath string contained in the cohort dataframe
            ('data', 'cache_embedding'),
            ('data', 'compute_text_embeddings'),
            ('model', 'encoder_ckpt'),
            ('model', 'embed_dim'),
            ('model', 'transformer_width'),
            ('model', 'classifier_depth'),
            # ('trainer', 'half_precision'),
            ('trainer', 'encoder_lr'),
            ('trainer', 'val_metric'),
            ('trainer', 'warmup'),
            ('trainer', 'early_stop'),
            ('trainer', 'predict_as_onehot'),
            ('trainer', 'regression'),
            ('trainer', 'lr_schedule'),
        ]

        for p1, p2 in optional_params:
            if p2 in config[p1].keys():
                setattr(self, p2, config[p1][p2])
            else:
                setattr(self, p2, None)
    
    @staticmethod
    def load_config(configf):
        with open(os.path.join(D.CONFIG_DIR, configf), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    
    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

def setup_logging(save_dir, log_config=os.path.join(D.SRC_DIR, "logger_config.json"), default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = os.path.join(save_dir, handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
        