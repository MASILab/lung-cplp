import os
import pandas as pd, numpy as np
import torch
import itertools
from pathlib import Path
from typing import TypedDict
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from lungcplp.utils import format_datetime_str
from tqdm import tqdm
import lungcplp.cachedcohorts as COHORTS

LLMs = {
    "stella_en_400M_v5":"dunzhang/stella_en_400M_v5",
    "NV-Embed-v2":"nvidia/NV-Embed-v2"
}

class Tab2LangEncoder():
    def __init__(self, 
                dataset,
                model,
                out_dir,
                batch_size=10,
                scan_level=False,
                classes=[0,1]):
        """
        Embeds tabular data into language space using a pre-trained language model.
            scan_level: if True, save embeddings at the scan level using format {pid}time{scandate}.npy, else patient level as {pid}.npy
        """
        self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=6, persistent_workers=True)
        self.model = SentenceTransformer(model, trust_remote_code=True, device="cuda")
        self.out_dir = out_dir
        self.scan_level = scan_level
        self.classes = classes
    
    def encode_save(self,):
        for batch in tqdm(self.dataloader, total=len(self.dataloader)):
            pid, scandate, data = batch['pid'], batch['scandate'], batch['data']
            embeddings = self.model.encode(data)

            for i in range(len(pid)):
                if self.scan_level:
                    np.save(os.path.join(self.out_dir, f"{pid[i]}time{scandate[i]}.npy"), embeddings[i])
                else:
                    np.save(os.path.join(self.out_dir, f"{pid[i]}.npy"), embeddings[i])
                    
    def encode_classes_save(self,):
        for batch in tqdm(self.dataloader, total=len(self.dataloader)):
            pid, scandate, data = batch['pid'], batch['scandate'], batch['data']
            data = list(itertools.chain(*data)) # flatten [[class0_0, ..., class0_n], [class1_0, ..., class1_n]] -> [class0_0, ..., class0_n, class1_0, ..., class1_n]
            embeddings = self.model.encode(data)
            
            for i in range(len(pid)):
                pid_dir = os.path.join(self.out_dir, pid[i])
                Path(pid_dir).mkdir(parents=True, exist_ok=True)
                for j in range(len(self.classes)):
                    idx = j*len(pid) + i
                    np.save(os.path.join(pid_dir, f"class{str(self.classes[j])}.npy"), embeddings[idx])
                    
    def encode_batch(self, batch):
        data = batch['data']
        embeddings = self.model.encode(data)
        return embeddings

class Item(TypedDict):
    pid: str
    scandate: str
    data: list

class Tab2JsonDataset(Dataset):
    """
    Converts tabular data to json format for language model embedding.
    Renames columns for better semantics
    """
    def __init__(self,
            df, # set scan_level=False if cohort grouped by pid
            datacache,
            scan_level=False, # if True, include scan-level variables, otherwise leave them out
            date_format: str="%Y%m%d",
            include_label=False,
        ):
        self.scan_level = scan_level
        if scan_level:
            self.df = df
            self.varset_cols = datacache.varset_scalar + datacache.varset_categorical + datacache.varset_scan
        else:
            self.df = df.loc[df.groupby('pid')['scanorder'].idxmax()]
            self.varset_cols = datacache.varset_scalar + datacache.varset_categorical
        self.labelset = datacache.labelset
        self.date_format = date_format
        self.include_label=include_label
        
        self.semantic_map = {
            'gender': 'Gender (1=M, 0=F)',
            'bmi': 'Body Mass Index (kg/m^2)',
            'race': 'Race (1=White, 2=Black, 3=Hispanic, 4=Asian, 5=American Indian or Alaskan Native, 6=Native Hawaiian or Other Pacific Islander, 0=Mixed, other, missing, unkown, or declined to answer)',
            'copd': 'Chronic Obstructive Pulmonary Disease',
            'emphysema': 'Emphysema (1=present, 0=absent)',
            'phist': 'History of any cancer',
            'fhist': 'Family history of lung cancer',
            'smo_status': 'Smoking status (1=current, 0=former)',
            'smo_duration': 'Smoking duration (years)',
            'smo_intensity': 'Smoking intensity (cigarettes/day)',
            'quit_time': 'Time since quitting smoking (years)',
            'pkyr': 'Smoking pack years',
            'cyfra': 'Serum concentration of hs-CYFRA 21-1 (in natural log of ng/mL)',
            'lc_subtype': 'Lung cancer histological subtype',
            'fup_days': 'Follow-up days',
        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        pid, scandate = row['pid'], format_datetime_str(row['scandate'], format=self.date_format)
        if self.include_label:
            varset = row[self.varset_cols + self.labelset]
        else:
            varset = row[self.varset_cols]
        varset = varset.rename(self.semantic_map)
        data = varset.to_json()
        return Item(pid=pid, scandate=scandate, data=data)

class Tab2LangDataset(Dataset):
    """
    Converts tabular data to language template for LLM embedding.
    """
    def __init__(self,
            df, # set scan_level=False if cohort grouped by pid
            datacache,
            scan_level=False, # if True, include scan-level variables, otherwise leave them out
            date_format: str="%Y%m%d",
            include_label=False,
            label="lung_cancer"
        ):
        self.scan_level = scan_level
        if scan_level:
            self.df = df
            self.varset_cols = datacache.varset_scalar + datacache.varset_categorical + datacache.varset_scan
        else:
            self.df = df.loc[df.groupby('pid')['scanorder'].idxmax()]
            self.varset_cols = datacache.varset_scalar + datacache.varset_categorical
        self.date_format = date_format
        self.label = label
        self.include_label=include_label
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        pid, scandate = row['pid'], format_datetime_str(row['scandate'], format=self.date_format)
        varset = row[self.varset_cols]
        lang_varset = [getattr(self, var)(value) for var, value in varset.items() if pd.notnull(value)]
        
        # insert lung cancer label
        if self.include_label and pd.notnull(row[self.label]):
            try:
                if pd.notnull(row[self.label]) and pd.notnull(row['lc_fup_days']):
                    lang_varset.append(self.lung_cancer_fup(row[self.label], row['lc_fup_days']))
                else:
                    lang_varset.append(self.lung_cancer(row[self.label]))
                if pd.notnull(row['lc_subtype']):
                    lang_varset.append(self.subtype(row['lc_subtype']))
            except KeyError:
                lang_varset.append(self.lung_cancer(row[self.label]))
        
        lang = " ".join(lang_varset)

        return Item(pid=pid, scandate=scandate, data=lang)
    
    def age(self, x):
        return f"The age of the subject is {int(x)}."

    def bmi(self, x):
        return f"The BMI of the subject is {x:.2f}."

    def quit_time(self, x):
        return f"The subject quit smoking {int(x)} years ago."

    def smo_duration(self, x):
        return f"The subject has been smoking for {int(x)} years."

    def smo_intensity(self, x):
        return f"The subject smokes {x:.2f} cigarettes per day."

    def pkyr(self, x):
        return f"The subject has {x:.2f} smoking pack years."

    def gender(self, x):
        x = "male" if x==1 else "female"
        return f"The sex of the subject is {x}."

    def race(self, x):
        map = {
            1:"White", 
            2:"Black",
            3:"Hispanic",
            4:"Asian",
            5:"American Indian or Alaskan Native",
            6:"Native Hawaiian or Other Pacific Islander",
            0:"Mixed, other, missing, unkown, or declined to answer",
        }
        return f"The race of the subject is {map[int(x)]}."

    def copd(self, x):
        x = "has" if x==1 else "does not have"
        return f"The subject {x} a history of Chronic Obstructive Pulmonary Disease."

    def emphysema(self, x):
        x = "has" if x==1 else "does not have"
        return f"The subject {x} Emphysema."

    def phist(self, x):
        x = "has" if x==1 else "does not have"
        return f"The subject {x} a history of any cancer."

    def fhist(self, x):
        x = "has" if x==1 else "does not have"
        return f"The subject {x} a family history of lung cancer."

    def smo_status(self, x):
        x = "is" if x==1 else "is not"
        return f"The subject {x} a current smoker."
    
    def cyfra(self, x):
        return f"The subject's serum concentration of hs-CYFRA 21-1 is {x:.2f} (in natural log of ng/mL)."
    
    def lung_cancer_fup(self, lung_cancer, fup_days):
        if lung_cancer==1:
            if fup_days >= 0:
                return f"The subject was diagnosed with lung cancer within {int(fup_days)} days."
            else:
                return f"The subject was diagnosed with lung cancer {int(abs(fup_days))} days ago."
        if lung_cancer==0:
            return f"The subject was not diagnosed with lung cancer."
        
    def lung_cancer(self, x):
        x = "was" if x==1 else "was not"
        return f"The subject {x} diagnosed with lung cancer."
    
    def subtype(self, x):
        return f"The histological subtype of the lung cancer is {x}."

class ClassesTab2LangDataset(Tab2LangDataset):
    def __init__(self,
            df, # set scan_level=False if cohort grouped by pid
            datacache,
            class_label="lung_cancer",
        ):
        super().__init__(df, datacache)
        self.df = df.loc[df.groupby('pid')['scanorder'].idxmax()]
        self.varset_cols = datacache.varset_scalar + datacache.varset_categorical
        self.classes = list(range(len(df[class_label].unique())))
        if class_label == "cancer_year1":
            self.generate_label_fn = lambda x: self.lung_cancer_fup(x, 750)
        else:
            self.generate_label_fn = lambda x: self.lung_cancer(x)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        pid, scandate = row['pid'], format_datetime_str(row['scandate'], format=self.date_format)
        varset = row[self.varset_cols]
        lang_varset = [getattr(self, var)(value) for var, value in varset.items() if pd.notnull(value)]
        
        # insert label for each class
        class_varsets = []
        for i in self.classes:
            class_varset = lang_varset.copy()
            class_varset.append(self.lung_cancer(i)) # i=(0,1)
            class_varsets.append(class_varset)
        
        lang = [" ".join(c) for c in class_varsets]
        return Item(pid=pid, scandate=scandate, data=lang)
            
def save_embeddings(cohort, model, batch_size=500, include_label=False, scan_level=False, label="lung_cancer"):
    cohort, subcohort = cohort.split('.')
    cohortwrapper = COHORTS.__dict__[cohort]()
    cohort_df = getattr(cohortwrapper, subcohort)
    drive="./data"
    
    if include_label:
        out_dir = os.path.join(cohortwrapper.cache.var_label_embeddings[model](drive))
    else:
        out_dir = os.path.join(cohortwrapper.cache.var_embeddings[model](drive))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    dataset = Tab2LangDataset(cohort_df, cohortwrapper.cache, scan_level=scan_level, include_label=include_label, label=label)
    encoder = Tab2LangEncoder(dataset, LLMs[model], out_dir, batch_size=batch_size, scan_level=scan_level)
    encoder.encode_save()
    
def save_json_embeddings(cohort, model, batch_size=500, include_label=False, scan_level=False, local_drive=True):
    cohort, subcohort = cohort.split('.')
    cohortwrapper = COHORTS.__dict__[cohort]()
    cohort_df = getattr(cohortwrapper, subcohort)
    drive = "./data"
    
    out_dir = os.path.join(cohortwrapper.cache.var_json_embeddings[model](drive))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    dataset = Tab2JsonDataset(cohort_df, cohortwrapper.cache, scan_level=scan_level, include_label=include_label)
    encoder = Tab2LangEncoder(dataset, LLMs[model], out_dir, batch_size=batch_size, scan_level=scan_level)
    encoder.encode_save()

def save_class_embeddings(cohort, model, batch_size=500, class_label="lung_cancer", local_drive=True):
    """
    Generate all classes for each patient, compute its embeddings, and save
    """
    cohort, subcohort = cohort.split('.')
    cohortwrapper = COHORTS.__dict__[cohort]()
    cohort_df = getattr(cohortwrapper, subcohort)
    drive = "./data"
    
    out_dir = os.path.join(cohortwrapper.cache.class_embeddings[model](drive))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    dataset = ClassesTab2LangDataset(cohort_df, cohortwrapper.cache, class_label=class_label)
    encoder = Tab2LangEncoder(dataset, LLMs[model], out_dir, batch_size=batch_size)
    encoder.encode_classes_save()
    

def embed_stats(dir):
    mu, sig = [], []
    for file in os.scandir(dir):
        embed = np.load(file)
        mu.append(embed.mean())
        sig.append(embed.std())
    return np.mean(mu), np.mean(sig)

if __name__ == '__main__':
    # Embed NLST patient-level variables with small model (dunzhang/stella_en_400M_v5)
    save_embeddings(
        cohort="NLST.train",
        model="stella_en_400M_v5",
        batch_size=10,
        include_label=False,
        scan_level=False,
    )
    