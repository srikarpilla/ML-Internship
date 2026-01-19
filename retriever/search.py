import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image

class FashionSearchEngine:
    
    
    def __init__(self, index_builder, encoder, metadata: dict):
        self.index_builder = index_builder
        self.encoder = encoder
        self.metadata = metadata
        self.image_paths = metadata['image_paths']
        
    def search(self, query: str, top_k: int = 10, use_compositional: bool = True) -> List[Dict]:
       
        print(f"\nSearching for: '{query}'")
        
        if use_compositional:
            query_embedding = self.encoder.encode_compositional_query(query)
        else:
            query_embedding = self.encoder.encode_text([query])
        
        distances, indices = self.index_builder.search(query_embedding, k=top_k)
        
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
            if idx < len(self.image_paths):
                img_path = self.image_paths[idx]
                
                result = {
                    'rank': rank,
                    'score': float(score),
                    'image_path': img_path,
                    'filename': Path(img_path).name,
                    'image_id': idx
                }
                
                
                if str(idx) in self.metadata.get('metadata', {}):
                    result.update(self.metadata['metadata'][str(idx)])
                
                results.append(result)
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 10) -> Dict[str, List[Dict]]:
       
        results = {}
        
        for query in queries:
            results[query] = self.search(query, top_k)
        
        return results
    
    def display_results(self, results: List[Dict], show_images: bool = True, 
                       max_display: int = 5):
        
        print(f"\nFound {len(results)} results:")
        print("-" * 80)
        
        for result in results[:max_display]:
            print(f"Rank {result['rank']}: {result['filename']}")
            print(f"  Score: {result['score']:.4f}")
            print(f"  Path: {result['image_path']}")
            
            if 'width' in result:
                print(f"  Size: {result['width']}x{result['height']}")
            
            print()
        
        if len(results) > max_display:
            print(f"... and {len(results) - max_display} more results")
    
    def get_result_images(self, results: List[Dict], max_images: int = 10) -> List[Image.Image]:
        
        images = []
        
        for result in results[:max_images]:
            try:
                img = Image.open(result['image_path'])
                images.append(img)
            except Exception as e:
                print(f"Error loading {result['image_path']}: {e}")
        
        return images
    
    def evaluate_query(self, query: str, ground_truth_ids: List[int], 
                       top_k: int = 10) -> Dict[str, float]:
       
        results = self.search(query, top_k=top_k)
        retrieved_ids = [r['image_id'] for r in results]
        
        relevant_retrieved = len(set(retrieved_ids) & set(ground_truth_ids))
        
        precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
        recall = relevant_retrieved / len(ground_truth_ids) if ground_truth_ids else 0
        
       
        ap = 0
        relevant_count = 0
        for i, img_id in enumerate(retrieved_ids, 1):
            if img_id in ground_truth_ids:
                relevant_count += 1
                ap += relevant_count / i
        
        ap = ap / len(ground_truth_ids) if ground_truth_ids else 0
        
        return {
            'precision@k': precision,
            'recall@k': recall,
            'average_precision': ap,
            'f1_score': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        }