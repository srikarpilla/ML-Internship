

Project Title: Multimodal Fashion & Context Retrieval System

Overview This project is an intelligent fashion search engine developed for the Glance ML Internship Assignment. It allows users to retrieve specific clothing images from a database using complex natural language descriptions, such as "a red tie and white shirt in a formal setting" or "casual weekend outfit in a park".


The Problem Solved Standard AI models (like "vanilla CLIP") often struggle with compositionality—for example, they might confuse a "blue shirt with red pants" for a "red shirt with blue pants". This system solves that issue by understanding the specific relationship between attributes (colors, items) and the context (location).


Technical Approach The system uses an Attribute-Conditioned Retrieval architecture:


Intelligent Encoding: It parses user queries into components (Color, Item, Context) and uses OpenCLIP to create a weighted search vector.


Fast Search: It utilizes FAISS (Facebook AI Similarity Search) to instantly search through image embeddings.


Precision Re-ranking: It applies a secondary filter using OpenCV to analyze pixel histograms, ensuring that retrieved images strictly match the requested colors.

User Interface The entire pipeline is wrapped in an interactive Streamlit web application, allowing users to upload datasets, type queries, and view retrieved images with similarity scores in real-time.
