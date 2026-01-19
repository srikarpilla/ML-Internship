import faiss
import numpy as np
from pathlib import Path
import pickle
from typing import Tuple

class FAISSIndexBuilder:
    
    
    def __init__(self, embedding_dim: int, index_type: str = "Flat"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        
    def build_flat_index(self, features: np.ndarray) -> faiss.IndexFlatIP:
       
        index = faiss.IndexFlatIP(self.embedding_dim)
        
       
        features = np.ascontiguousarray(features.astype('float32'))
        
        print(f"Adding {features.shape[0]} vectors to flat index...")
        index.add(features)
        
        print(f"Index built. Total vectors: {index.ntotal}")
        return index
    
    def build_ivf_index(self, features: np.ndarray, nlist: int = 100) -> faiss.IndexIVFFlat:
        
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        
        
        index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        features = np.ascontiguousarray(features.astype('float32'))
        
        print(f"Training IVF index with {nlist} clusters...")
        index.train(features)
        
        print(f"Adding {features.shape[0]} vectors to IVF index...")
        index.add(features)
        
        print(f"Index built. Total vectors: {index.ntotal}")
        return index
    
    def build(self, features: np.ndarray, nlist: int = 100) -> faiss.Index:
     
        n_samples = features.shape[0]
        
       
        if self.index_type == "IVF" and n_samples > 1000:
    
            optimal_nlist = min(nlist, int(np.sqrt(n_samples)))
            self.index = self.build_ivf_index(features, optimal_nlist)
        else:
            self.index = self.build_flat_index(features)
        
        return self.index
    
    def set_search_params(self, nprobe: int = 10):
        
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = nprobe
            print(f"Set nprobe to {nprobe}")
    
    def save_index(self, save_path: Path):
       
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_path))
        print(f"Index saved to {save_path}")
    
    def load_index(self, load_path: Path) -> faiss.Index:
        
        self.index = faiss.read_index(str(load_path))
        print(f"Index loaded from {load_path}. Total vectors: {self.index.ntotal}")
        return self.index
    
    def search(self, query_features: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
       
        if self.index is None:
            raise ValueError("No index loaded. Build or load index first.")
        
        query_features = np.ascontiguousarray(query_features.astype('float32'))
        
       
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        
        distances, indices = self.index.search(query_features, k)
        return distances, indices
    
    def get_stats(self) -> dict:
     
        if self.index is None:
            return {}
        
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.embedding_dim,
            'index_type': type(self.index).__name__
        }
        
        if isinstance(self.index, faiss.IndexIVFFlat):
            stats['nlist'] = self.index.nlist
            stats['nprobe'] = self.index.nprobe
        
        return stats