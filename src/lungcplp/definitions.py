import os

from dotenv import load_dotenv
load_dotenv()

# Paths
ROOT_DIR = "./lung-cplp"
LOCAL_DIR = "./lung-clip"
DATA_DIR = os.path.join(ROOT_DIR, "data")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
RUNS_DIR = os.path.join(LOCAL_DIR, "runs")
SRC_DIR = os.path.join(ROOT_DIR, "src/lung-cplp")

# SEED
# RANDOM_SEED = 1193
RANDOM_SEED = 1105