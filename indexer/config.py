import os
import sys
from pathlib import Path

class Config:
    
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    TRAIN_IMG_DIR = RAW_DATA_DIR / "train"
    VAL_IMG_DIR = RAW_DATA_DIR / "val_test"
    
    INDEX_DIR = PROCESSED_DATA_DIR / "index"
    METADATA_FILE = PROCESSED_DATA_DIR / "metadata.pkl"
    
    
    CLIP_MODEL = "ViT-B/32"
    
    # Device detection that works on Windows without nvidia-smi
    @staticmethod
    def get_device():
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except:
            pass
        return "cpu"
    
    DEVICE = get_device.__func__()  
    BATCH_SIZE = 32
    IMG_SIZE = 224
    
  
    EMBEDDING_DIM = 512
    INDEX_TYPE = "IVF"  
    NLIST = 100  
    NPROBE = 10  
    
   
    MAX_IMAGES = 1000
    MIN_IMAGES = 500
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    
    TOP_K = 10
    RERANK_TOP_K = 50
    
    @classmethod
    def ensure_dirs(cls):
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.RAW_DATA_DIR.mkdir(exist_ok=True)
        cls.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        cls.INDEX_DIR.mkdir(exist_ok=True)