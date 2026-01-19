

import streamlit as st
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import time
from typing import List, Dict, Tuple
import sys


try:
    import torch
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    st.error("PyTorch or OpenCLIP not installed. Please run: pip install torch open-clip-torch")
    st.stop()

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    st.error("FAISS not installed. Please run: pip install faiss-cpu")
    st.stop()

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    





st.set_page_config(
    page_title="Fashion Search",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)


#  CSS


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .search-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
    }
    
    .result-card {
        background: white;
        border-radius: 1rem;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 1rem;
        border: 2px solid transparent;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    
    .score-excellent {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 1.5rem;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .score-good {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 1.5rem;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .score-moderate {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 1.5rem;
        font-weight: 600;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 2rem;
        font-size: 1.1rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# FASHION ENCODER CLASS


class FashionEncoder:
    """Enhanced CLIP encoder for fashion retrieval"""
    
    def __init__(self, model_name="ViT-B-32", device="cpu"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
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
    
    def encode_images(self, images: List, batch_size: int = 32) -> np.ndarray:
        """Encode images to feature vectors"""
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
        """Encode text queries to feature vectors"""
        text_tokens = self.tokenizer(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def encode_compositional_query(self, query: str) -> np.ndarray:
        """Enhanced encoding for compositional queries"""
        main_encoding = self.encode_text([query])
        
        query_lower = query.lower()
        
        # Detect attributes
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 
                  'purple', 'pink', 'brown', 'gray', 'grey']
        detected_colors = [c for c in colors if c in query_lower]
        
        items = ['shirt', 'pants', 'jacket', 'coat', 'dress', 'skirt', 'tie', 
                 'blazer', 'hoodie', 't-shirt', 'jeans', 'suit']
        detected_items = [item for item in items if item in query_lower]
        
        locations = ['office', 'park', 'street', 'home', 'city', 'indoor', 
                     'outdoor', 'beach', 'formal', 'casual']
        detected_locations = [loc for loc in locations if loc in query_lower]
        
        # Generate augmented queries
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





@st.cache_resource
def load_system():
    """Load the complete retrieval system"""
    
    # Paths
    base_path = Path(".")
    index_path = base_path / "data" / "processed" / "index" / "faiss_index.bin"
    metadata_path = base_path / "data" / "processed" / "metadata.pkl"
    
    # Checks  if files exist
    if not index_path.exists():
        return None, None, None, f"Index not found at {index_path}"
    
    if not metadata_path.exists():
        return None, None, None, f"Metadata not found at {metadata_path}"
    
    try:
        # Loads metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = FashionEncoder(device=device)
        
        index = faiss.read_index(str(index_path))
        
        return encoder, index, metadata, None
    
    except Exception as e:
        return None, None, None, str(e)

def search_images(encoder, index, metadata, query: str, top_k: int = 10, 
                 use_compositional: bool = True) -> List[Dict]:
    
    
   
    if use_compositional:
        query_embedding = encoder.encode_compositional_query(query)
    else:
        query_embedding = encoder.encode_text([query])
    
    query_embedding = np.ascontiguousarray(query_embedding.astype('float32'))
    
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    image_paths = metadata['image_paths']
    
    for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
        if idx < len(image_paths):
            img_path = image_paths[idx]
            
            result = {
                'rank': rank,
                'score': float(score),
                'image_path': img_path,
                'filename': Path(img_path).name,
                'image_id': idx
            }
            
            results.append(result)
    
    return results

def rerank_by_color(query: str, results: List[Dict], top_k: int = 10) -> List[Dict]:
    """Re-rank results based on color matching"""
    
    if not CV2_AVAILABLE:
        return results
    
    query_lower = query.lower()
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 
              'orange', 'purple', 'pink', 'brown', 'gray', 'grey']
    detected_colors = [c for c in colors if c in query_lower]
    
    if not detected_colors:
        return results
    
    # Color ranges in RGB
    color_map = {
        'red': ([200, 255], [0, 100], [0, 100]),
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
    
    
    for result in results:
        try:
            img = cv2.imread(result['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            color_scores = []
            for color in detected_colors:
                if color in color_map:
                    r_range, g_range, b_range = color_map[color]
                    
                    mask = (
                        (img[:,:,0] >= r_range[0]) & (img[:,:,0] <= r_range[1]) &
                        (img[:,:,1] >= g_range[0]) & (img[:,:,1] <= g_range[1]) &
                        (img[:,:,2] >= b_range[0]) & (img[:,:,2] <= b_range[1])
                    )
                    
                    proportion = mask.sum() / (img.shape[0] * img.shape[1])
                    color_scores.append(proportion)
            
            color_score = np.mean(color_scores) if color_scores else 0.5
            result['reranked_score'] = 0.7 * result['score'] + 0.3 * color_score
            result['color_match'] = color_score
        except:
            result['reranked_score'] = result['score']
            result['color_match'] = 0.5
    
    
    results.sort(key=lambda x: x['reranked_score'], reverse=True)
    
    for rank, result in enumerate(results[:top_k], 1):
        result['rank'] = rank
    
    return results[:top_k]

def get_score_class(score: float) -> Tuple[str, str]:
    """Get CSS class and label based on score"""
    if score >= 0.28:
        return "score-excellent", "🔥 Excellent"
    elif score >= 0.24:
        return "score-good", "✨ Good"
    else:
        return "score-moderate", "👍 Moderate"


def main():
   
    st.markdown('<div class="main-title">👗 Fashion Search AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Find fashion items using natural language</div>', unsafe_allow_html=True)
    
    with st.spinner("Loading AI models..."):
        encoder, index, metadata, error = load_system()
    
    if error:
        st.error(f"❌ Error: {error}")
        st.info("Please ensure you have run: `python run_indexer.py` first")
        st.stop()
    
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        top_k = st.slider(
            "Number of results",
            min_value=3,
            max_value=24,
            value=9,
            step=3,
            help="How many images to retrieve"
        )
        
        grid_cols = st.slider(
            "Grid columns",
            min_value=2,
            max_value=5,
            value=3,
            help="Images per row"
        )
        
        use_rerank = st.checkbox(
            "Enable color re-ranking",
            value=True,
            help="Better results for color queries"
        )
        
        use_compositional = st.checkbox(
            "Compositional encoding",
            value=True,
            help="Better for multi-attribute queries"
        )
        
        st.markdown("---")
        
        
        st.markdown("### 📊 System Info")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{len(metadata['image_paths'])}</div>
                <div class="metric-label">Images</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            device = "CUDA" if torch.cuda.is_available() else "CPU"
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{device}</div>
                <div class="metric-label">Device</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        
        st.markdown("### 💡 Examples")
        
        examples = {
            "🌧️ Yellow Raincoat": "A person in a bright yellow raincoat",
            "💼 Business Attire": "Professional business attire inside a modern office",
            "👕 Blue Shirt Park": "Someone wearing a blue shirt sitting on a park bench",
            "👟 Casual Weekend": "Casual weekend outfit for a city walk",
            "👔 Red Tie Formal": "A red tie and a white shirt in a formal setting"
        }
        
        for label, query_text in examples.items():
            if st.button(label, key=f"ex_{label}", use_container_width=True):
                st.session_state.query = query_text
    
    
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
   
    default_query = st.session_state.get('query', '')
    query = st.text_input(
        "🔍 What are you looking for?",
        value=default_query,
        placeholder="e.g., person wearing a blue jacket in a park",
        label_visibility="collapsed",
        key="search_input"
    )
    
    if 'query' in st.session_state:
        del st.session_state.query
    
   
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("🚀 Search Now", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
  
    if search_button and query.strip():
        with st.spinner(f"🔍 Searching for '{query}'..."):
            start_time = time.time()
            
            results = search_images(
                encoder, 
                index, 
                metadata, 
                query, 
                top_k=top_k,
                use_compositional=use_compositional
            )
            
            
            if use_rerank and results:
                colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
                if any(c in query.lower() for c in colors):
                    results = rerank_by_color(query, results, top_k)
            
            search_time = time.time() - start_time
        
      
        if results:
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ Results Found", len(results))
            with col2:
                st.metric("⚡ Search Time", f"{search_time:.2f}s")
            with col3:
                avg_score = sum(r['score'] for r in results) / len(results)
                st.metric("📊 Avg Score", f"{avg_score:.3f}")
            
            st.markdown("---")
            st.markdown("### 📸 Results")
            
          
            num_rows = (len(results) + grid_cols - 1) // grid_cols
            
            for row in range(num_rows):
                cols = st.columns(grid_cols)
                
                for col_idx in range(grid_cols):
                    result_idx = row * grid_cols + col_idx
                    
                    if result_idx < len(results):
                        result = results[result_idx]
                        
                        with cols[col_idx]:
                            try:
                                
                                img = Image.open(result['image_path'])
                                st.image(img, use_column_width=True)
                                
                                
                                score_class, score_label = get_score_class(result['score'])
                                
                              
                                st.markdown(f"""
                                <div class="result-card">
                                    <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">
                                        🏆 Rank #{result['rank']}
                                    </div>
                                    <div class="{score_class}">
                                        {score_label}: {result['score']:.3f}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                               
                                with open(result['image_path'], 'rb') as f:
                                    st.download_button(
                                        label="💾 Download",
                                        data=f.read(),
                                        file_name=result['filename'],
                                        mime="image/jpeg",
                                        key=f"dl_{result_idx}",
                                        use_container_width=True
                                    )
                            
                            except Exception as e:
                                st.error(f"Error: {e}")
        else:
            st.error("❌ No results found. Try different keywords!")
    
    elif search_button:
        st.warning("⚠️ Please enter a search query")
    
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #888;">
        <p style="font-size: 0.9rem;">Built with Streamlit + OpenCLIP + FAISS</p>
        <p style="font-size: 0.8rem;">Fashion Retrieval System with Compositional Understanding</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()