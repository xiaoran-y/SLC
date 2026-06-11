# Datasets

This directory should contain the preprocessed datasets used in the paper.
Due to licensing restrictions, we do not redistribute the raw data.
Please download and preprocess each dataset as described below.

## Knowledge Tracing Datasets

| Dataset | Source | Preprocessing |
|---------|--------|---------------|
| `assist2017_pid_uid_time_pos/` | [ASSISTments 2017](https://sites.google.com/view/assistmentsdatamining/dataset) | Temporal split (train/valid/test) with pid, uid, count columns |
| `assist2009_pid_uid_time/` | [ASSISTments 2009](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) | Temporal split with pid, uid, count columns |
| `algebra_merged_pid_uid_time/` | [KDD Cup 2005-2009](https://pslcdatashop.web.cmu.edu/KDDCup/) | Merged 2005-06, 2006-07, 2008-09 releases; temporal split |
| `eedi_task12_pid_uid_time/` | [Eedi NeurIPS 2020](https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip) | Tasks 1+2 merged; temporal split |

## Cross-Domain Control Datasets

| Dataset | Source | Preprocessing |
|---------|--------|---------------|
| `flight_delay/` | [Flight Delay 2018-2022 (Kaggle)](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022) | 2018 train, 2019 calib/test; see `preprocess/preprocess_flight_delay.py` |
| `ml1m_pid_uid_time/` | [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) | Temporal split with user/item/count |

## Data Format

Each KT dataset directory contains:
- `meta.json` — dataset metadata (`n_question`, `n_pid`, `n_users`, `seqlen`)
- `<dataset>_train1.csv`, `<dataset>_valid1.csv`, `<dataset>_test1.csv` — temporal splits

CSV format (4 lines per student sequence):
```
line 0: student_id, true_student_id, base_offset
line 1: problem_ids (comma-separated)
line 2: question/skill_ids (comma-separated)
line 3: responses (comma-separated, 0/1)
```

`base_offset` is the cumulative interaction count before this split, used to compute the global time index `count = base_offset + position`.
