# Contrastive Patient-level Pretraining Enables Longitudinal and Multimodal Fusion for Lung Cancer Risk Prediction

[[Paper]](https://openreview.net/pdf?id=cyHmr0DIjM)

Pretrained Model:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14816443.svg)](https://doi.org/10.5281/zenodo.14816443)



## Setup
Dependencies: 
* Pytorch 2.6.0
* packages in `requirements.txt`
* an editable installation of the project itself: `pip install -e ./lung-cplp`
* Download pretrained CLIP model if not training from scratch.
* Reproducing paper results requires public data from the [NLST](https://www.cancerimagingarchive.net/collection/nlst/)
* Feature vectors from [sybil](https://github.com/reginabarzilaygroup/Sybil)

Organize public data into the following clinical variables

| Column Name   | Description |
|--------------|------------|
| `pid`        | UID for each patient. |
| `scandate`   | The date when the scan was performed (YYYY-MM-DD). |
| `scanorder`  | int: order of scans for a patient with 0 being earliest |
| `age`        | patient age at time of randomization |
| `bmi`        | body mass index at time of randomization |
| `gender`     | 1=Male, 0=Female |
| `race`       | 0=Mixed, other, missing, unkown, or declined to answer, 1=White, 2=Black, 3=Hispanic, 4=Asian, 5=American Indian or Alaskan Native, 6=Native Hawaiian or Other Pacific Islander |
| `quit_time`  | int: years since the patient quit smoking |
| `smo_duration` | total years the patient has smoked |
| `smo_intensity` | number of cigarettes smoked per day |
| `pkyr`       | packs per day × years smoked |
| `copd`       | Whether the patient has Chronic Obstructive Pulmonary Disease (1=Yes/0=No). |
| `emphysema`  | Whether the patient has emphysema (1=Yes/0=No) |
| `phist`      | Personal history of any type of cancer (1=Yes/0=No) |
| `fhist`      | Family history of lung cancer (1=Yes/0=No) |
| `smo_status` | 0=Current Smoker, 1=Former Smoker |


## Usage
Ensure clinical variables are organized in `cachedcohorts.py` and compute text embeddings using `embeddings.py`

### Training CLIP

Training from random weights
```
python cli.py clip_nlst.yaml pretrain
```
To resume trainig from checkpoint, indicate checkpoint in config file
```yaml
trainer:
    resume: "./epoch=1.pth"
```

### Zeroshot/Finetuning CLIP

Config .yaml should point to pretrained model
```yaml
...
model:
  encoder_ckpt: "./model_best-epoch=9002-loss=0.022.pth"
```
```
python cli.py 0shot_clip_nlst.yaml zeroshot     # zeroshot
python cli.py ft_clip_nlst.yaml train           # finetuning
```

### Testing
```
python cli.py ft_clip_nlst.yaml test --cohort NLST.test --checkpoint best_ft_model.pth  # test
python cli.py ft_clip_nlst.yaml bootstrap --cohort NLST.test                            # bootstrap mean AUC and 95% CIs
```

## Cite
```
@inproceedings{
li2025contrastive,
title={Contrastive Patient-level Pretraining Enables Longitudinal and Multimodal Fusion for Lung Cancer Risk Prediction},
author={Thomas Li},
booktitle={Submitted to Medical Imaging with Deep Learning},
year={2025},
url={https://openreview.net/forum?id=cyHmr0DIjM},
note={under review}
}
```


