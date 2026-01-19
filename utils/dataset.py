import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm

class FashionDataset:
    def __init__(self, img_dirs: List[Path], max_images: int = None):
        self.img_dirs = img_dirs
        self.max_images = max_images
        self.image_paths = []
        self.metadata = {}
        
    def load_images(self, supported_formats: set) -> List[Path]:
        """Scan directories and collect image paths"""
        print("Scanning directories for images...")
        
        for img_dir in self.img_dirs:
            if not img_dir.exists():
                print(f"Warning: Directory {img_dir} does not exist")
                continue
                
            for root, _, files in os.walk(img_dir):
                for fname in files:
                    if Path(fname).suffix in supported_formats:
                        self.image_paths.append(Path(root) / fname)
                        
                        if self.max_images and len(self.image_paths) >= self.max_images:
                            break
                if self.max_images and len(self.image_paths) >= self.max_images:
                    break
        
        print(f"Found {len(self.image_paths)} images")
        return self.image_paths
    
    def extract_metadata(self):
        """Extract basic metadata from images"""
        print("Extracting metadata...")
        
        for idx, img_path in enumerate(tqdm(self.image_paths)):
            try:
                img = Image.open(img_path)
                
                self.metadata[idx] = {
                    'path': str(img_path),
                    'filename': img_path.name,
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'size_kb': img_path.stat().st_size / 1024
                }
                img.close()
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        return self.metadata
    
    def save_metadata(self, filepath: Path):
        """Save metadata to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'image_paths': [str(p) for p in self.image_paths],
                'metadata': self.metadata
            }, f)
        print(f"Metadata saved to {filepath}")
    
    @staticmethod
    def load_metadata(filepath: Path) -> Dict:
        """Load metadata from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded metadata for {len(data['image_paths'])} images")
        return data
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        if not self.metadata:
            return {}
            
        widths = [m['width'] for m in self.metadata.values()]
        heights = [m['height'] for m in self.metadata.values()]
        sizes = [m['size_kb'] for m in self.metadata.values()]
        
        return {
            'total_images': len(self.metadata),
            'avg_width': np.mean(widths),
            'avg_height': np.mean(heights),
            'avg_size_kb': np.mean(sizes),
            'total_size_mb': sum(sizes) / 1024
        }