

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from indexer.config import Config
from utils.dataset import FashionDataset
from models.fashion_encoder import FashionEncoder
from indexer.extract_features import FeatureExtractor
from indexer.build_index import FAISSIndexBuilder

def main(args):
    print("=" * 80)
    print("Fashion Retrieval System - Indexing Pipeline")
    print("=" * 80)
    
    Config.ensure_dirs()
    
    print("\n[1/5] Loading dataset...")
    img_dirs = []
    
    if Config.TRAIN_IMG_DIR.exists():
        img_dirs.append(Config.TRAIN_IMG_DIR)
    if Config.VAL_IMG_DIR.exists():
        img_dirs.append(Config.VAL_IMG_DIR)
    
    if not img_dirs:
        print("ERROR: No image directories found!")
        print(f"Expected locations:")
        print(f"  - {Config.TRAIN_IMG_DIR}")
        print(f"  - {Config.VAL_IMG_DIR}")
        print("\nPlease unzip your Fashionpedia images to these directories.")
        return
    
    dataset = FashionDataset(img_dirs, max_images=Config.MAX_IMAGES)
    image_paths = dataset.load_images(Config.SUPPORTED_FORMATS)
    
    if len(image_paths) < Config.MIN_IMAGES:
        print(f"WARNING: Found only {len(image_paths)} images.")
        print(f"Minimum recommended: {Config.MIN_IMAGES}")
    
    dataset.extract_metadata()
    dataset.save_metadata(Config.METADATA_FILE)
    
    stats = dataset.get_stats()
    print(f"\nDataset statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")
    
    
    print("\n[2/5] Initializing encoder...")
    encoder = FashionEncoder(
        model_name=Config.CLIP_MODEL,
        pretrained="openai",
        device=Config.DEVICE
    )
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.CLIP_MODEL}")
    print(f"Embedding dimension: {encoder.get_embedding_dim()}")
    
    
    print("\n[3/5] Extracting features...")
    feature_extractor = FeatureExtractor(encoder, batch_size=Config.BATCH_SIZE)
    
    features_path = Config.PROCESSED_DATA_DIR / "features.pkl"
    features = feature_extractor.extract_all(image_paths, save_path=features_path)
    
    if features.size == 0:
        print("ERROR: Failed to extract features!")
        return
    
    print("\n[4/5] Building FAISS index...")
    index_builder = FAISSIndexBuilder(
        embedding_dim=encoder.get_embedding_dim(),
        index_type=Config.INDEX_TYPE
    )
    
    index = index_builder.build(features, nlist=Config.NLIST)
    
    index_builder.set_search_params(nprobe=Config.NPROBE)
    
   
    print("\n[5/5] Saving index...")
    index_path = Config.INDEX_DIR / "faiss_index.bin"
    index_builder.save_index(index_path)
    
    
    print("\n" + "=" * 80)
    print("Indexing Complete!")
    print("=" * 80)
    print(f"\nIndex statistics:")
    for key, val in index_builder.get_stats().items():
        print(f"  {key}: {val}")
    
    print(f"\nFiles saved:")
    print(f"  Metadata: {Config.METADATA_FILE}")
    print(f"  Features: {features_path}")
    print(f"  Index: {index_path}")
    
    print("\nYou can now run the retrieval pipeline with:")
    print("  python run_retriever.py \"your search query\"")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build fashion retrieval index")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum number of images to index")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for feature extraction")
    
    args = parser.parse_args()
    
    if args.max_images:
        Config.MAX_IMAGES = args.max_images
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    
    main(args)