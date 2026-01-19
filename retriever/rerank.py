import numpy as np
from typing import List, Dict
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Color-based re-ranking will be disabled.")

class ResultReranker:
   
    
    def __init__(self, encoder):
        self.encoder = encoder
        
    def extract_color_histogram(self, img_path: str, bins: int = 32) -> np.ndarray:
        
        if not CV2_AVAILABLE:
            return np.zeros(bins * 3)
            
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            hist_r = cv2.calcHist([img], [0], None, [bins], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [bins], [0, 256])
            hist_b = cv2.calcHist([img], [2], None, [bins], [0, 256])
            
            hist = np.concatenate([hist_r, hist_g, hist_b])
            hist = hist / (hist.sum() + 1e-7)
            
            return hist.flatten()
        except:
            return np.zeros(bins * 3)
    
    def compute_color_similarity(self, query_colors: List[str], img_path: str) -> float:
        
        if not query_colors:
            return 0.5
        
        color_map = {
            'red': ([200, 255], [0, 100], [0, 100]),      # RGB ranges
            'blue': ([0, 100], [0, 100], [200, 255]),
            'green': ([0, 100], [200, 255], [0, 100]),
            'yellow': ([200, 255], [200, 255], [0, 100]),
            'black': ([0, 50], [0, 50], [0, 50]),
            'white': ([200, 255], [200, 255], [200, 255]),
            'orange': ([200, 255], [100, 200], [0, 100]),
            'purple': ([100, 200], [0, 100], [100, 200]),
            'pink': ([200, 255], [100, 200], [150, 255]),
            'brown': ([100, 150], [50, 100], [0, 50]),
            'gray': ([100, 150], [100, 150], [100, 150]),
            'grey': ([100, 150], [100, 150], [100, 150])
        }
        
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            scores = []
            for color in query_colors:
                if color.lower() in color_map:
                    r_range, g_range, b_range = color_map[color.lower()]
                    
                    mask = (
                        (img[:,:,0] >= r_range[0]) & (img[:,:,0] <= r_range[1]) &
                        (img[:,:,1] >= g_range[0]) & (img[:,:,1] <= g_range[1]) &
                        (img[:,:,2] >= b_range[0]) & (img[:,:,2] <= b_range[1])
                    )
                    
                   
                    proportion = mask.sum() / (img.shape[0] * img.shape[1])
                    scores.append(proportion)
            
            return np.mean(scores) if scores else 0.5
        except:
            return 0.5
    
    def rerank_by_attributes(self, query: str, results: List[Dict], 
                            top_k: int = 10) -> List[Dict]:
       
        query_lower = query.lower()
        
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 
                  'orange', 'purple', 'pink', 'brown', 'gray', 'grey']
        detected_colors = [c for c in colors if c in query_lower]
        
        for result in results:
            img_path = result['image_path']
            
            base_score = result['score']
           
            color_score = 0
            if detected_colors:
                color_score = self.compute_color_similarity(detected_colors, img_path)
            
            
            result['reranked_score'] = 0.7 * base_score + 0.3 * color_score
            result['color_match_score'] = color_score
        
        
        results.sort(key=lambda x: x['reranked_score'], reverse=True)
        
       
        for rank, result in enumerate(results[:top_k], 1):
            result['rank'] = rank
        
        return results[:top_k]
    
    def rerank_by_diversity(self, results: List[Dict], top_k: int = 10, 
                           diversity_weight: float = 0.3) -> List[Dict]:
       
        if len(results) <= 1:
            return results
     
        features = []
        for result in results:
            img = Image.open(result['image_path'])
            feat = self.encoder.encode_images([img])[0]
            features.append(feat)
        
        features = np.array(features)
        
        
        selected_indices = [0]  
        selected = [results[0]]
        
        while len(selected) < min(top_k, len(results)):
            max_min_dist = -1
            best_idx = -1
            
            for i, result in enumerate(results):
                if i in selected_indices:
                    continue
                
                
                min_dist = min([
                    1 - np.dot(features[i], features[j]) 
                    for j in selected_indices
                ])
                
              
                score = (1 - diversity_weight) * result['score'] + diversity_weight * min_dist
                
                if score > max_min_dist:
                    max_min_dist = score
                    best_idx = i
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
                selected.append(results[best_idx])
        
        
        for rank, result in enumerate(selected, 1):
            result['rank'] = rank
        
        return selected