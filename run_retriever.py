

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from indexer.config import Config
from models.fashion_encoder import FashionEncoder
from indexer.build_index import FAISSIndexBuilder
from retriever.search import FashionSearchEngine
from retriever.rerank import ResultReranker
from utils.dataset import FashionDataset

def visualize_results(results, max_display=5):
    """Display search results in a grid"""
    n_results = min(len(results), max_display)
    
    if n_results == 0:
        print("No results to display")
        return
    
    fig, axes = plt.subplots(1, n_results, figsize=(4*n_results, 4))
    
    if n_results == 1:
        axes = [axes]
    
    for idx, (ax, result) in enumerate(zip(axes, results[:n_results])):
        try:
            img = Image.open(result['image_path'])
            ax.imshow(img)
            ax.set_title(f"Rank {result['rank']}\nScore: {result['score']:.3f}", 
                        fontsize=10)
            ax.axis('off')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading image", 
                   ha='center', va='center')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('search_results.png', dpi=150, bbox_inches='tight')
    print(f"\nResults visualization saved to: search_results.png")
    plt.close()

def run_evaluation_queries(search_engine, reranker):
    """Run the 5 evaluation queries from the assignment"""
    
    queries = [
        "A person in a bright yellow raincoat",
        "Professional business attire inside a modern office",
        "Someone wearing a blue shirt sitting on a park bench",
        "Casual weekend outfit for a city walk",
        "A red tie and a white shirt in a formal setting"
    ]
    
    print("\n" + "=" * 80)
    print("Running Evaluation Queries")
    print("=" * 80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}/5] {query}")
        print("-" * 80)
        
        results = search_engine.search(query, top_k=10, use_compositional=True)
        
        
        if any(word in query.lower() for word in ['red', 'blue', 'yellow', 'green', 'white', 'black']):
            print("Applying color-based re-ranking...")
            results = reranker.rerank_by_attributes(query, results, top_k=10)
        
        search_engine.display_results(results, max_display=3)
        
       
        visualize_results(results, max_display=min(5, len(results)))
        print(f"Visualization saved as: search_results.png")

def main(args):
    print("=" * 80)
    print("Fashion Retrieval System - Query Interface")
    print("=" * 80)
    
    
    index_path = Config.INDEX_DIR / "faiss_index.bin"
    if not index_path.exists():
        print("\nERROR: Index not found!")
        print(f"Expected location: {index_path}")
        print("\nPlease run the indexer first:")
        print("  python run_indexer.py")
        return
    
    
    print("\n[1/4] Loading metadata...")
    if not Config.METADATA_FILE.exists():
        print(f"ERROR: Metadata file not found at {Config.METADATA_FILE}")
        return
    
    metadata = FashionDataset.load_metadata(Config.METADATA_FILE)
    
    
    print("\n[2/4] Loading encoder...")
    encoder = FashionEncoder(
        model_name=Config.CLIP_MODEL,
        pretrained="openai",
        device=Config.DEVICE
    )
    
    
    print("\n[3/4] Loading FAISS index...")
    index_builder = FAISSIndexBuilder(
        embedding_dim=encoder.get_embedding_dim(),
        index_type=Config.INDEX_TYPE
    )
    index_builder.load_index(index_path)
    index_builder.set_search_params(nprobe=Config.NPROBE)
    
    print(f"\nIndex stats:")
    for key, val in index_builder.get_stats().items():
        print(f"  {key}: {val}")
    
    
    print("\n[4/4] Initializing search engine...")
    search_engine = FashionSearchEngine(index_builder, encoder, metadata)
    reranker = ResultReranker(encoder)
    
    print("\n" + "=" * 80)
    print("Ready to search!")
    print("=" * 80)
    
    
    if args.eval:
        run_evaluation_queries(search_engine, reranker)
    elif args.query:
       
        print(f"\nQuery: {args.query}")
        print("-" * 80)
        
        results = search_engine.search(
            args.query, 
            top_k=args.top_k,
            use_compositional=not args.no_compositional
        )
        
        if args.rerank:
            print("\nApplying re-ranking...")
            results = reranker.rerank_by_attributes(args.query, results, top_k=args.top_k)
        
        search_engine.display_results(results, max_display=args.display)
        
        if args.visualize:
            visualize_results(results, max_display=min(args.display, len(results)))
    else:
        
        print("\nInteractive search mode. Type 'quit' to exit.")
        print("Special commands:")
        print("  eval - Run all 5 evaluation queries")
        print("  quit - Exit the program")
        
        while True:
            try:
                query = input("\nEnter search query: ").strip()
                
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'eval':
                    run_evaluation_queries(search_engine, reranker)
                    continue
                elif not query:
                    continue
                
                results = search_engine.search(query, top_k=10, use_compositional=True)
                
                
                if any(word in query.lower() for word in ['red', 'blue', 'yellow', 'green', 'white', 'black']):
                    results = reranker.rerank_by_attributes(query, results, top_k=10)
                
                search_engine.display_results(results, max_display=5)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue
    
    print("\n" + "=" * 80)
    print("Search session ended")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search fashion dataset")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of results to return")
    parser.add_argument("--display", type=int, default=5,
                       help="Number of results to display")
    parser.add_argument("--eval", action="store_true",
                       help="Run evaluation queries")
    parser.add_argument("--rerank", action="store_true",
                       help="Apply re-ranking to results")
    parser.add_argument("--no-compositional", action="store_true",
                       help="Disable compositional query encoding")
    parser.add_argument("--visualize", action="store_true",
                       help="Save visualization of results")
    
    args = parser.parse_args()
    main(args)