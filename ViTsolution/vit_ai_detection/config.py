# config.py
import torch

class Config:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_LABELS = 2
    DATASET_NAME = "tiny-genimage"
    MERGED_CSV = "merged_dataset.csv"

    MODEL_TYPE = 'vit'
    FREEZE_BACKBONE = True

    EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    NUM_WORKERS = 4
    PIN_MEMORY = True

    LOG_DIR = "logs"
    CHECKPOINT_DIR = "checkpoints"

    EARLY_STOPPING_PATIENCE = 6
    EARLY_STOPPING_MIN_DELTA = 1e-4

    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-6

    USE_AMP = True


    import os
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    DEBUG = False

    TRANSFER_LEARNING = False
    TRANSFER_MODEL_PATH = ""
    TRANSFER_NUM_CLASSES = 0

