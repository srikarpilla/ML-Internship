import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image
import pickle

class FeatureExtractor:
    
    def __init__(self, encoder, batch_size: int = 32):
        self.encoder = encoder
        self.batch_size = batch_size
        self.features = None
        
    def extract_batch(self, image_paths: List[Path], start_idx: int = 0) -> Tuple[np.ndarray, List[int]]:
      
        valid_images = []
        valid_indices = []
        
        for idx, img_path in enumerate(image_paths, start=start_idx):
            try:
                img = Image.open(img_path).convert('RGB')
                valid_images.append(img)
                valid_indices.append(idx)
                
                if len(valid_images) >= self.batch_size:
                    break
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
                continue
        
        if not valid_images:
            return None, []
        
        features = self.encoder.encode_images(valid_images, batch_size=len(valid_images))
        return features, valid_indices
    
    def extract_all(self, image_paths: List[Path], save_path: Path = None) -> np.ndarray:
        """Extract features from all images with progress tracking"""
        all_features = []
        feature_map = {}  
        
        print(f"Extracting features from {len(image_paths)} images...")
        
        batch_start = 0
        pbar = tqdm(total=len(image_paths), desc="Feature extraction")
        
        while batch_start < len(image_paths):
            batch_end = min(batch_start + self.batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            
            features, valid_indices = self.extract_batch(batch_paths, start_idx=batch_start)
            
            if features is not None:
                all_features.append(features)
                
               
                for orig_idx in valid_indices:
                    feature_map[orig_idx] = len(all_features) - 1
            
            batch_start = batch_end
            pbar.update(len(batch_paths))
        
        pbar.close()
        
        self.features = np.vstack(all_features) if all_features else np.array([])
        
        print(f"Extracted {self.features.shape[0]} feature vectors of dimension {self.features.shape[1]}")
        
        if save_path:
            self.save_features(save_path, feature_map)
        
        return self.features
    
    def save_features(self, save_path: Path, feature_map: dict = None):
       
        save_data = {
            'features': self.features,
            'feature_map': feature_map,
            'shape': self.features.shape
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Features saved to {save_path}")
    
    @staticmethod
    def load_features(load_path: Path) -> Tuple[np.ndarray, dict]:
      
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded features of shape {data['shape']}")
        return data['features'], data.get('feature_map', {})