import os
from functools import cached_property
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from dataclasses import dataclass
from collections.abc import Callable
from typing import Optional
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from lungcplp.utils import read_tabular, format_datetime_str, todays
from cohorts.cli import NAMES, NLSTCohort, LIVUCohort, LIVUSPNCohort, MCLCohort, BronchCohort, VLSPCohort, NoduleVUCohort

import lungcplp.definitions as D

@dataclass
class CachedCohort:
    name: str
    init: Callable
    varset_categorical: Optional[list] = None
    varset_scalar: Optional[list] = None
    varset_scan: Optional[list] = None
    labelset: Optional[list] = None
    var_embeddings: Optional[dict] = None
    var_label_embeddings: Optional[dict] = None
    var_json_embeddings: Optional[dict] = None
    class_embeddings: Optional[dict] = None
    sybil: Optional[Callable] = None # pretrained sybil features

SYBIL_STATS = {'mean': 0.26725134, 'std': 0.7492173}

NLST_CACHE = CachedCohort(
    name=NAMES.nlst,
    init=lambda: NLSTCohort(
            cache="./nlst.csv",
            scan_cache="./nlst_scan.csv",
        ),
    varset_scalar=['age','bmi','quit_time','smo_duration','smo_intensity','pkyr'],
    varset_categorical=['gender', 'race', 'copd', 'emphysema', 'phist', 'fhist', 'smo_status'],
    varset_scan=['slice_thickness', 'nodule_size', 'nodule_count', 'nodule_attenuation', 'upper_lobe', 'nodule_spiculation'],
    labelset = ['lung_cancer', 'fup_days', 'lc_subtype'],
    var_embeddings = {
            "stella_en_400M_v5": lambda d: os.path.join(d, "nlst/tab2lang/stella_en_400M_v5"),
        },
    var_label_embeddings = {
            "stella_en_400M_v5": lambda d: os.path.join(d, "nlst/tab2lang_wlabels/stella_en_400M_v5"),
        },
    var_json_embeddings = {
        "stella_en_400M_v5": lambda d: os.path.join(d, "nlst/tab2json/stella_en_400M_v5"),
    },
    class_embeddings = {
        "stella_en_400M_v5": lambda d: os.path.join(d, "nlst/tab2lang_classes/stella_en_400M_v5"),
    },
    sybil=lambda d: os.path.join(d, "nlst/Sybil/all"),
)

######### Cohort Wrappers

class CohortWrapper():
    def __init__(
        self,
        cache: CachedCohort,
        label: str, **kwargs,
    ):
        self.cache = cache
        self.label = label

    @cached_property
    def cohort(self):
        return self.cache.init()

    @cached_property
    def scan_cohort_df(self):
        df = self.cohort.scan_cohort_df
        # df = df[df['lung_cancer'].notnull()]
        df = df[df['slice_thickness'] < 5]
        df = df[df['scandate'].notnull()]
        return df

######### NLST

class NLST(CohortWrapper):
    def __init__(
        self,
        cache=NLST_CACHE,
        label="cancer_year1",
        fup_label = "scan_fup_days",
        test='./ardila_test_set.xlsx',
        **kwargs,
    ):
        super().__init__(cache, label)
        self.label = label
        self.fup_label = fup_label
        self.test_path = test
        self.imputer = IterativeImputer(max_iter=10, random_state=D.RANDOM_SEED)
        
    @cached_property
    def scan_cohort_df(self):
        df = self.cohort.scan_cohort_df
        df = df[df['scandate'].notnull()]
        df = df.groupby(['pid', 'scandate'], as_index=False).max()
        df['scandate'] = pd.to_datetime(df['scandate'], errors='coerce')
        maxdate = df.loc[df.groupby('pid')['scandate'].idxmax()]
        maxdate = maxdate.rename(columns={'scandate': 'maxdate'})[['pid', 'maxdate']]
        df = df.merge(maxdate, on=['pid'], how='left')
        df['shifted_scandate'] = df['maxdate'] - pd.to_timedelta(df['duration'], unit='D')
        df['shifted_scandate'] = df['shifted_scandate'].apply(format_datetime_str)
        df['scandate'] = df['scandate'].apply(lambda x: format_datetime_str(x, format="%Y"))
        # gender variabel (1,2) -> (0,1)
        df['gender'] = df['gender'] - 1
        df['lc_fup_days'] = df['scan_fup_days']
        return df
    
    @cached_property
    def cs(self):
        df = self.scan_cohort_df
        return df.loc[df.groupby('pid')['scanorder'].idxmax()]
    
    @cached_property
    def train(self):
        df = self.scan_cohort_df
        df = df.groupby(['pid', 'scandate'], as_index=False).first() # if multiple scans within a scandate, take a random one
        return df[~df['pid'].isin(self.test['pid'])]
    
    @cached_property
    def test(self):
        test = read_tabular(self.test_path, {'dtype':{'patient_id':str}})
        # df = self.scan_cohort_df[self.scan_cohort_df['lung_cancer'].notnull()] # comment out if unconfirmed
        scan = self.scan_cohort_df.groupby(['pid', 'scandate'], as_index=False).first()
        scan['study_yr'] = (scan.groupby('pid')['scandate'].rank(method='dense', ascending=True) - 1).astype(int)
        test = scan.merge(test, left_on=['pid', 'study_yr'], right_on=['patient_id', 'study_yr'], how='inner')
        return test.groupby(['pid', 'study_yr'], as_index=False).max() # duplicate rows
    
    @cached_property
    def imputed(self):
        varset_df = self.scan_cohort_df[self.cache.varset_scalar + self.cache.varset_categorical]
        id_df = self.scan_cohort_df[['pid', 'scandate', 'scanorder', 'shifted_scandate', 'lung_cancer', self.label, self.fup_label]]
        imputed = pd.DataFrame(self.imputer.fit_transform(varset_df), columns=varset_df.columns, index=varset_df.index)
        df = id_df.merge(imputed, left_index=True, right_index=True)
        return df

    @cached_property
    def imputed_train(self):
        df = self.imputed
        df = df[df['pid'].isin(self.train['pid'])]
        return df
    
    
    @cached_property
    def imputed_test(self):
        df = self.imputed
        df = df[df['pid'].isin(self.test['pid'])]
        return df
    
    # Nodules ==================================
    @cached_property
    def test_nodules(self):
        df = self.test
        wnodule = df.loc[df.groupby('pid')['scanorder'].idxmax()]
        wnodule = wnodule[wnodule['nodule_count'].notnull()]
        return df[df['pid'].isin(wnodule['pid'])]
    
    @cached_property
    def imputed_test_nodules(self):
        df = self.imputed_test
        df = df.merge(self.scan_cohort_df[['pid', 'scandate', 'nodule_count']], on=['pid', 'scandate',])
        wnodule = df.loc[df.groupby('pid')['scanorder'].idxmax()]
        wnodule = wnodule[wnodule['nodule_count'].notnull()]
        return df[df['pid'].isin(wnodule['pid'])]

