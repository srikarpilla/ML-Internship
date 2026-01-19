import torch
import torch.nn as nn
import open_clip
from PIL import Image
import numpy as np
from typing import List, Union

class FashionEncoder:
    
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(device)
        self.model.eval()
        
        
        self.attribute_templates = {
            'color': "a photo of {color} colored clothing",
            'clothing': "a photo of a person wearing {item}",
            'context': "a photo taken in {location}",
            'style': "a photo of {style} fashion style"
        }
        
    def encode_images(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> np.ndarray:
        
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_imgs = []
            
            for img in batch:
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                batch_imgs.append(self.preprocess(img))
            
            batch_tensor = torch.stack(batch_imgs).to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            
            all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
       
        text_tokens = self.tokenizer(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def encode_compositional_query(self, query: str) -> np.ndarray:
       
        main_encoding = self.encode_text([query])
        
        query_lower = query.lower()
        
       
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 
                  'purple', 'pink', 'brown', 'gray', 'grey']
        detected_colors = [c for c in colors if c in query_lower]
        
      
        items = ['shirt', 'pants', 'jacket', 'coat', 'dress', 'skirt', 'tie', 
                 'blazer', 'hoodie', 't-shirt', 'jeans', 'suit']
        detected_items = [item for item in items if item in query_lower]
        
        locations = ['office', 'park', 'street', 'home', 'city', 'indoor', 
                     'outdoor', 'beach', 'formal', 'casual']
        detected_locations = [loc for loc in locations if loc in query_lower]
        
       
        augmented_queries = [query]
        
        for color in detected_colors:
            augmented_queries.append(self.attribute_templates['color'].format(color=color))
        
        for item in detected_items:
            augmented_queries.append(self.attribute_templates['clothing'].format(item=item))
        
        for loc in detected_locations:
            if loc in ['formal', 'casual']:
                augmented_queries.append(self.attribute_templates['style'].format(style=loc))
            else:
                augmented_queries.append(self.attribute_templates['context'].format(location=loc))
        
        
        if len(augmented_queries) > 1:
            all_encodings = self.encode_text(augmented_queries)
          
            weights = np.array([2.0] + [1.0] * (len(augmented_queries) - 1))
            weights = weights / weights.sum()
            combined = np.average(all_encodings, axis=0, weights=weights)
            combined = combined / np.linalg.norm(combined)
            return combined.reshape(1, -1)
        
        return main_encoding
    
    def get_embedding_dim(self) -> int:
       
        return self.model.visual.output_dim