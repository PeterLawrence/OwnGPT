# -*- coding: utf-8 -*-
"""
Create ownChat  private gpt
@author: Lawrence P.J.
"""
import os
persist_directory = "." + os.sep + "chroma_db"
source_directory = "." + os.sep + "source_documents"
embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"
collection_name = "collection_name"
model_path = "." + os.sep + "Models" + os.sep + "zephyr-7b-beta.Q4_0.gguf"
model_n_ctx = 5000
model_type = "LlamaCpp"
#model_type = "GPT4All"

# Data processing chunk size and overlap
# a larger resolution of 512 or 1024 is possible
# with an overlap around 50
# see https://www.pinecone.io/learn/chunking-strategies/
chunk_size = 256 
chunk_overlap = 20
