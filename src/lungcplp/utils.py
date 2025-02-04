import json, os, math, re, random, datetime
import numpy as np, pandas as pd
import datetime
from collections import OrderedDict
from pathlib import Path
from itertools import repeat
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch.optim as optim
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from sklearn.metrics import roc_auc_score

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
        
def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader
        
def plot_feature_importance(weights, out_dir):
    # plots feature importance given vector of weights
    # returns feature indices in descending order of importance and their corresponding weights
    importance = np.squeeze(abs(weights))
    feature_ranking = importance.argsort()[::-1] # descending order
    importance = importance[feature_ranking] # sort by rank
    
    p10 = round(0.1*len(importance))
    top10p = importance[:p10] # top 10% of features
    plt.figure(figsize=(10,10))
    plt.bar(range(len(top10p)), top10p, tick_label=feature_ranking[:p10])
    plt.xlabel("Feature index")
    plt.ylabel("Abs feature weight")
    plt.savefig(os.path.join(out_dir, "feature_importance.png"))
    return feature_ranking, importance

def todays(x):
    if pd.notnull(x):
        return x.days
    else:
        return None

def cosine_annealing_warmup_restarts(optimizer, config, total_training_steps, **kwargs):
    return CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=math.ceil(total_training_steps/10),
        max_lr=config.lr,
        min_lr=config.lr/100,
        warmup_steps=math.ceil(total_training_steps/50),
    )

def one_cycle_lr(optimizer, config, total_training_steps, **kwargs):
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_training_steps,
        pct_start=0.3,
        anneal_strategy='cos',
    )


def find_best_checkpoint(ckpt_path, metric="max val/auc"):
    mode, metric = metric.split()
    assert mode in ['min', 'max']
    best_metric = np.inf if mode == 'min' else -np.inf
    best_checkpoint = None

    # Regex to extract the test_metric from the file name
    metric = str(metric).replace('/', '_')
    pattern = re.compile(fr'model_best-epoch=\d+-{metric}=([\d\.]+)\.pth')

    for file in os.listdir(ckpt_path):
        match = pattern.match(file)
        if match:
            test_metric = float(match.group(1))
            if mode == 'max':
                if test_metric > best_metric:
                    best_metric = test_metric
                    best_checkpoint = os.path.join(ckpt_path, file)
            else:
                if test_metric < best_metric:
                    best_metric = test_metric
                    best_checkpoint = os.path.join(ckpt_path, file)

    return best_checkpoint

def unpack_batch(batch, config, device):
    # returns expected model input from batch
    if config.dataset == 'Imgft_Dataset':
        img = batch['data'].to(device)
    else:
        img, expr = batch['data']
        img, expr = img.to(device), expr.to(device)
    times, padding, label = batch['times'].to(device), batch['padding'].to(device), batch['label'].to(device)
    
    if config.model_name=='tdvit':
        x = (img, padding, times)
    elif config.model_name=='img_classifier':
        x = (img,)
    else:
        x = (img, expr, padding, times)
    
    return x, label

def auc_from_df(df, cols=('label', 'prob')):
    y_col, yhat_col = cols
    return roc_auc_score(df[y_col], df[yhat_col])

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
            "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def read_tabular(fpath, fargs={}):
    suffix = os.path.basename(fpath).split('.')[-1]
    if suffix == 'csv':
        return pd.read_csv(fpath, **fargs)
    elif suffix in ['xlsx', 'xlsm']:
        return pd.read_excel(fpath, **fargs, engine='openpyxl')
    elif suffix == 'xls':
        return pd.read_excel(fpath, **fargs)
    raise NotImplementedError(f"Unsupported file type: {fpath}")

def format_datetime_str(x, format="%Y%m%d") -> str:
    if pd.isnull(x):
        return None
    elif isinstance(x, str):
        dt = pd.to_datetime(x, errors='coerce')
    elif isinstance(x, float) or isinstance(x, int):
        dt = pd.to_datetime(str(int(x)), errors='coerce')
    elif isinstance(x, datetime):
        dt = x
    else:
        dt = pd.to_datetime(str(x), errors='coerce')
    
    if pd.notnull(dt): # check if NaT
        return dt.strftime(format)
    else:
        return None

def is_nested_list(l):
    try:
        next(x for x in l if isinstance(x, (list, tuple)))
        return True
    except StopIteration:
        return False

def ci(data, confidence=0.95):
    d = 1.0*np.array(data)
    n = len(d)
    mu, std = np.mean(d), np.std(d)
    z = stats.norm.ppf(1-(1-confidence)/2)
    me = z*std/math.sqrt(n)
    return mu, mu-me, mu+me

def bootstrap(df, agg, grps=[], n=100, confidence=0.95):
    """
    Compute mean and CI from n bootstrap samples, sampling with replacement.
    Per central limit theorem, sample means are normally distributed.

    Parameters
    ----------
    df: pandas.DataFrame. metrics in long format
    agg: func. method of computing the aggregate metric (i.e. mean, AUC, F1score, etc.)
    *grps: str or list. name of col to group by
    n: int. number of bootstrap samples
    confidence: float.

    Returns
    ----------
    bstrap: pandas.DataFrame. mean metrics and CI grouped by grps
    """
    grps = grps if isinstance(grps, list) else [grps]
    grpnames = [df[g].unique().tolist() for g in grps]
    grpcomb = itertools.product(*grpnames) # all combinations of the groups
    resultrows = []

    for comb in grpcomb:
        if comb: # if grp exists
            query = ' & '.join(f"{g}=={c}" for g, c in zip(grps, comb)) # str with 'col1==grp1 & col2==grp2 & ...'
            dfgrp = df.query(query) # get the group
        else:
            dfgrp = df

        # compute aggregate metric on n samples
        metrics = []
        for i in range(n):
            sample = dfgrp.sample(frac=1.0, replace=True)
            metrics.append(agg(sample))
        
        metrics = list(zip(*metrics)) if is_nested_list(metrics) else [metrics]

        # get mean and CIs for each metric type
        cis = [] # [(mu1, lci_1, uci_1), (mu2, lci_2, uci_2), ...]
        for m in metrics:
            cis.append(ci(m, confidence=confidence))
        cidict = [{f'mean_{i}': c[0], f'lci_{i}': c[1], f'uci_{i}': c[2]} for i, c in enumerate(cis)]
        cidict = {k: v for dict in cidict for k, v in dict.items()}
        grpdict = {k[0]: k[1] for k in zip(grps, comb)} if comb else {}
        resultrows.append({**grpdict, **cidict})
    
    return pd.DataFrame(resultrows)
