import os, pandas as pd, numpy as np
from typing import TypedDict
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from tlungcplp.utils import format_datetime_str
from lungcplp.cachedcohorts import SYBIL_STATS
import lungcplp.cachedcohorts as COHORTS, lungcplp.datasets as DATASETS

import lungcplp.definitions as D

@dataclass
class Phase:
    pretrain: str="pretrain"
    train: str = "train"
    finetune: str = "finetune"
    test: str="test"
    finetune_test: str="finetune_test"
    predict: str="predict"
    zeroshot: str="zeroshot"
    bootstrap: str="bootstrap"

PHASE = Phase()

class Item(TypedDict):
    pid: str
    times: torch.Tensor
    data: list
    label: torch.Tensor
    mods: torch.Tensor
    
    
class DataModule():
    def __init__(self, config, phase, dataroot, subset=1.0):
        self.batch_size = config.batch_size
        
        label = config.label if config.label is not None else "lung_cancer"
        cohort, subcohort = config.cohort.split('.')
        cohortwrapper = COHORTS.__dict__[cohort](label=label, dataroot=dataroot, cache_embedding=config.cache_embedding)
        cohort_df = getattr(cohortwrapper, subcohort)
        cohort_df = cohort_df.sample(frac=subset)
        
        print(f"n scans: {len(cohort_df)}")
        print(f"n subjects: {len(cohort_df['pid'].unique())}")
        print(cohort_df[label].value_counts())
        
        if config.val_split == 0:
            train_items, val_items = cohort_df, cohort_df
        else:
            cs = cohort_df.loc[cohort_df.groupby('pid')[label].idxmax()]
            val_items = cs.groupby(label, group_keys=False).apply(lambda x: x.sample(frac=config.val_split))
            val_items = cohort_df[cohort_df['pid'].isin(val_items['pid'])]
            train_items = cohort_df.drop(val_items.index)
        
        if phase in [PHASE.train, PHASE.pretrain, PHASE.finetune]:
            self.train_dataset = DATASETS.__dict__[config.dataset](
                df=train_items,
                datacache=cohortwrapper.cache,
                phase=phase,
                i_dim=config.img_dim,
                s_dim=config.expr_dim,
                slen=config.slen,
                label=label,
                date_format=config.date_format,
                dataroot=dataroot,
                fpath_from_df=config.fpath_from_df or False,
                cache_embedding=config.cache_embedding or "var_embeddings",
                # half_precision=half_precision,
            )
            self.val_dataset = DATASETS.__dict__[config.dataset](
                df=val_items,
                datacache=cohortwrapper.cache,
                phase=phase,
                i_dim=config.img_dim,
                s_dim=config.expr_dim,
                slen=config.slen,
                label=label,
                date_format=config.date_format,
                dataroot=dataroot,
                fpath_from_df=config.fpath_from_df or False,
                cache_embedding=config.cache_embedding or "var_embeddings",
                # half_precision=half_precision,
            )

        if phase in [PHASE.test, PHASE.predict, PHASE.zeroshot]:
            self.test_dataset = DATASETS.__dict__[config.dataset](
                df=cohort_df,
                datacache=cohortwrapper.cache,
                phase=phase,
                i_dim=config.img_dim,
                s_dim=config.expr_dim,
                slen=config.slen,
                label=label,
                date_format=config.date_format,
                dataroot=dataroot,
                fpath_from_df=config.fpath_from_df or False,
                cache_embedding=config.cache_embedding or "var_embeddings",
                # half_precision=half_precision,
            )
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=6, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=6, persistent_workers=True, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=3)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=6, persistent_workers=True)
    
    
class ZScoreNorm:
    """
    mean, std: int or array
        if int, then input is normalized globally
        if array, then input is normalized per channel. array dim must equal input dim
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std
    
class ToTensor:
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
    def __call__(self, x):
        return torch.tensor(x, dtype=self.dtype)

class Imgft_Lang_Dataset(Dataset):
    def __init__(self,
                df,
                datacache,
                phase: str=PHASE.pretrain,
                i_dim: int=512,
                s_dim: int=12,
                slen: int=3,
                label: str="lung_cancer",
                date_format: str="%Y%m%d",
                # half_precision: bool=True,
                dataroot: str="./data",
                llm: str="stella_en_400M_v5",
                fpath_from_df=False,
                cache_embedding="var_embedding", # [var_embedding, var_label_embedding, var_json_embedding]
                **kwargs
                ):
        # super().__init__(Dataset, **kwargs)
        super().__init__()
        self.df = df
        self.datacache = datacache
        self.phase = phase
        self.i_dim = i_dim
        self.s_dim = s_dim
        self.slen = slen
        self.label = label
        self.date_format = date_format
        self.dataroot = dataroot
        self.llm = llm
        self.fpath_from_df = fpath_from_df
        self.cache_embedding = getattr(datacache, cache_embedding)
        # self.precision = torch.float16 if half_precision else torch.float32
        self.precision = torch.float32
        self.pids = self.df['pid'].unique().tolist()

        self.img_transforms = [ToTensor(dtype=self.precision)]
        self.var_transforms = [ToTensor(dtype=self.precision)]
        
        
    def __getitem__(self, index) -> Item:
        pid = self.pids[index]
        rows = self.df[self.df['pid']==pid].sort_values(by='scanorder', ascending=False)
        rows = rows.iloc[:self.slen]
        n = min(self.slen, len(rows))
        
        padding = torch.zeros(self.slen, dtype=self.precision)
        padding[:len(rows)] = 1
        
        # load image features
        img_seq = np.zeros((self.slen, self.i_dim), dtype=np.float32)
        for i, (_, row) in enumerate(rows.iterrows()):
            scandate = format_datetime_str(row.scandate, format=self.date_format)
            if self.fpath_from_df:
                fpath = row.scan_fpath
            else:
                fpath = os.path.join(self.datacache.sybil(self.dataroot), f"{pid}time{scandate}.npy")
            img_seq[i] = np.load(fpath).flatten()
        for transform in self.img_transforms:
            img_seq = transform(img_seq)
        
        # tab2lang embeddings
        if self.fpath_from_df:
            fpath = row.embed_fpath
        else:
            fpath = os.path.join(self.cache_embedding[self.llm](self.dataroot), f"{pid}.npy")
        embed = np.load(fpath)
        embed = np.expand_dims(embed, axis=0) # (d,) -> (1, d)
        for transform in self.var_transforms:
            embed = transform(embed)
            
        # relative times in the format [img1, img2, ..., expr1, expr2, ...]
        reldays = torch.zeros(self.slen, dtype=self.precision)
        dates = rows['scandate'].apply(lambda x: pd.to_datetime(x))
        for i in range(n): # fill in first half of reldays
            reldays[i] = (dates.iloc[0] - dates.iloc[i]).days
        
        if self.phase in [PHASE.pretrain, PHASE.predict]:
            label = torch.zeros(1, dtype=torch.int64)
        else:
            label = rows.iloc[0][self.label]
            label = torch.zeros(1, dtype=torch.int64) if pd.isnull(label) else torch.tensor(label, dtype=torch.int64)
            
        item = Item(pid=pid,
                    times=reldays,
                    padding=padding,
                    data=[img_seq, embed],
                    label=label,)
        
        return item
    
    def __len__(self) -> int:
        return len(self.pids)

class Imgft_Tabular_Dataset(Dataset):
    def __init__(self,
                df,
                datacache,
                phase: str=PHASE.pretrain,
                i_dim: int=512,
                s_dim: int=12,
                slen: int=3,
                label: str="lung_cancer",
                date_format: str="%Y%m%d",
                # half_precision: bool=True,
                dataroot: str="./data",
                **kwargs
                ):
        # super().__init__(Dataset, **kwargs)
        super().__init__()
        self.df = df
        self.datacache = datacache
        self.phase = phase
        self.i_dim = i_dim
        self.s_dim = s_dim
        self.slen = slen
        self.label = label
        self.date_format = date_format
        self.dataroot = dataroot
        # self.precision = torch.float16 if half_precision else torch.float32
        self.precision = torch.float32
        self.pids = self.df['pid'].unique().tolist()
        
        mu, sig = SYBIL_STATS['mean'], SYBIL_STATS['std']
        self.img_transforms = [ZScoreNorm(mu, sig), ToTensor(dtype=self.precision)]
        
        varset_df = df[datacache.varset_scalar]
        varset_mu = varset_df.mean(axis=0).to_numpy(dtype=np.float32)
        varset_sig = varset_df.std(axis=0).to_numpy(dtype=np.float32)
        self.var_transforms = [ZScoreNorm(varset_mu, varset_sig)]
        
    def __getitem__(self, index) -> Item:
        pid = self.pids[index]
        rows = self.df[self.df['pid']==pid].sort_values(by='scanorder', ascending=False)
        rows = rows.iloc[:self.slen]
        n = min(self.slen, len(rows))
        
        padding = torch.zeros(self.slen, dtype=self.precision)
        padding[:len(rows)] = 1
        
        # load image features
        img_seq = np.zeros((self.slen, self.i_dim), dtype=np.float32)
        for i, (_, row) in enumerate(rows.iterrows()):
            scandate = format_datetime_str(row.scandate, format=self.date_format)
            img_seq[i] = np.load(os.path.join(self.datacache.sybil(self.dataroot), f"{pid}time{scandate}.npy")).flatten()
        for transform in self.img_transforms:
            img_seq = transform(img_seq)
        
        # clinical variables
        if len(self.datacache.varset_scalar) > 0:
            expr = rows.iloc[0][self.datacache.varset_scalar].to_numpy(dtype=np.float32)
            for transform in self.var_transforms:
                expr = transform(expr)
        if len(self.datacache.varset_categorical) > 0:
            expr = np.concatenate([expr, rows.iloc[0][self.datacache.varset_categorical].to_numpy(dtype=np.float32)])
        expr = np.expand_dims(expr, axis=0) # (d,) -> (1, d)
        expr = torch.tensor(expr, dtype=self.precision)
            
        # relative times in the format [img1, img2, ..., expr1, expr2, ...]
        reldays = torch.zeros(self.slen, dtype=self.precision)
        dates = rows['shifted_scandate'].apply(lambda x: pd.to_datetime(x))
        for i in range(n): # fill in first half of reldays
            reldays[i] = (dates.iloc[0] - dates.iloc[i]).days
        
        if self.phase in [PHASE.pretrain, PHASE.predict]:
            label = torch.zeros(1, dtype=torch.int64)
        else:
            label = rows.iloc[0][self.label]
            label = torch.zeros(1, dtype=torch.int64) if pd.isnull(label) else torch.tensor(label, dtype=torch.int64)
            
        item = Item(pid=pid,
                    times=reldays,
                    padding=padding,
                    data=[img_seq, expr],
                    label=label,)
        
        return item
    
    def __len__(self) -> int:
        return len(self.pids)

class ZeroShot_Imgft_Lang_Dataset(Dataset):
    def __init__(self,
                df,
                datacache,
                phase: str=PHASE.pretrain,
                i_dim: int=512,
                s_dim: int=12,
                slen: int=3,
                label: str="lung_cancer",
                date_format: str="%Y%m%d",
                # half_precision: bool=True,
                dataroot: str="./data",
                llm: str="stella_en_400M_v5",
                fpath_from_df=False,
                cache_embedding="var_embedding", # [var_embedding, var_label_embedding, var_json_embedding, zeroshot]
                **kwargs
                ):
        # super().__init__(Dataset, **kwargs)
        super().__init__()
        self.df = df
        self.datacache = datacache
        self.phase = phase
        self.i_dim = i_dim
        self.s_dim = s_dim
        self.slen = slen
        self.label = label
        self.date_format = date_format
        self.dataroot = dataroot
        self.llm = llm
        self.fpath_from_df = fpath_from_df
        self.cache_embedding = getattr(datacache, "class_embeddings")
        # self.precision = torch.float16 if half_precision else torch.float32
        self.precision = torch.float32
        self.pids = self.df['pid'].unique().tolist()

        self.img_transforms = [ToTensor(dtype=self.precision)]
        self.var_transforms = [ToTensor(dtype=self.precision)]
        
        
    def __getitem__(self, index) -> Item:
        pid = self.pids[index]
        rows = self.df[self.df['pid']==pid].sort_values(by='scanorder', ascending=False)
        rows = rows.iloc[:self.slen]
        n = min(self.slen, len(rows))
        
        padding = torch.zeros(self.slen, dtype=self.precision)
        padding[:len(rows)] = 1
        
        # load image features
        img_seq = np.zeros((self.slen, self.i_dim), dtype=np.float32)
        for i, (_, row) in enumerate(rows.iterrows()):
            scandate = format_datetime_str(row.scandate, format=self.date_format)
            if self.fpath_from_df:
                fpath = row.scan_fpath
            else:
                fpath = os.path.join(self.datacache.sybil(self.dataroot), f"{pid}time{scandate}.npy")
            img_seq[i] = np.load(fpath).flatten()
        for transform in self.img_transforms:
            img_seq = transform(img_seq)
        
        # tab2lang embeddings
        class0 = os.path.join(self.cache_embedding[self.llm](self.dataroot), f"{pid}", "class0.npy")
        class1 = os.path.join(self.cache_embedding[self.llm](self.dataroot), f"{pid}", "class1.npy")
        class0, class1 = np.load(class0), np.load(class1)
        embed = np.stack((class0, class1), axis=0)
        for transform in self.var_transforms:
            embed = transform(embed)
            
        # relative times in the format [img1, img2, ..., expr1, expr2, ...]
        reldays = torch.zeros(self.slen, dtype=self.precision)
        dates = rows['scandate'].apply(lambda x: pd.to_datetime(x))
        for i in range(n): # fill in first half of reldays
            reldays[i] = (dates.iloc[0] - dates.iloc[i]).days
        
        if self.phase in [PHASE.pretrain, PHASE.predict]:
            label = torch.zeros(1, dtype=torch.int64)
        else:
            label = rows.iloc[0][self.label]
            label = torch.zeros(1, dtype=torch.int64) if pd.isnull(label) else torch.tensor(label, dtype=torch.int64)
            
        item = Item(pid=pid,
                    times=reldays,
                    padding=padding,
                    data=[img_seq, embed],
                    label=label,)    
        return item
    
    def __len__(self) -> int:
        return len(self.pids)


class Imgft_Dataset(Dataset):
    def __init__(self,
                df,
                datacache,
                phase: str=PHASE.pretrain,
                i_dim: int=512,
                slen: int=3,
                label: str="lung_cancer",
                date_format: str="%Y%m%d",
                # half_precision: bool=True,
                dataroot: str="./data",
                **kwargs
                ):
        # super().__init__(Dataset, **kwargs)
        super().__init__()
        self.df = df
        self.datacache = datacache
        self.phase = phase
        self.i_dim = i_dim
        self.slen = slen
        self.label = label
        self.date_format = date_format
        self.dataroot = dataroot
        # self.precision = torch.float16 if half_precision else torch.float32
        self.precision = torch.float32
        self.pids = self.df['pid'].unique().tolist()
        
        mu, sig = SYBIL_STATS['mean'], SYBIL_STATS['std']
        self.img_transforms = [ZScoreNorm(mu, sig), ToTensor(dtype=self.precision)]

        
    def __getitem__(self, index) -> Item:
        pid = self.pids[index]
        rows = self.df[self.df['pid']==pid].sort_values(by='scanorder', ascending=False)
        rows = rows.iloc[:self.slen]
        n = min(self.slen, len(rows))
        
        padding = torch.zeros(self.slen, dtype=self.precision)
        padding[:len(rows)] = 1
        
        # load image features
        img_seq = np.zeros((self.slen, self.i_dim), dtype=np.float32)
        for i, (_, row) in enumerate(rows.iterrows()):
            scandate = format_datetime_str(row.scandate, format=self.date_format)
            img_seq[i] = np.load(os.path.join(self.datacache.sybil(self.dataroot), f"{pid}time{scandate}.npy")).flatten()
        for transform in self.img_transforms:
            img_seq = transform(img_seq)
        
            
        # relative times in the format [img1, img2, ..., expr1, expr2, ...]
        reldays = torch.zeros(self.slen, dtype=self.precision)
        dates = rows['shifted_scandate'].apply(lambda x: pd.to_datetime(x))
        for i in range(n): # fill in first half of reldays
            reldays[i] = (dates.iloc[0] - dates.iloc[i]).days
        
        if self.phase in [PHASE.pretrain, PHASE.predict]:
            label = torch.zeros(1, dtype=torch.int64)
        else:
            label = rows.iloc[0][self.label]
            label = torch.zeros(1, dtype=torch.int64) if pd.isnull(label) else torch.tensor(label, dtype=torch.int64)
            
        item = Item(pid=pid,
                    times=reldays,
                    padding=padding,
                    data=img_seq,
                    label=label,)
        
        return item
    
    def __len__(self) -> int:
        return len(self.pids)
    
    
    
