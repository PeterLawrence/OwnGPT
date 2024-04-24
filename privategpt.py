# -*- coding: utf-8 -*-
"""
Create ownChat test private gpt
@author: Lawrence P.J
"""

import chromadb
from chromadb.config import Settings

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp

import owngptsettings
import sys


def private_gpt_generate_msg(human_msg,verbose_output):
    embeddings = HuggingFaceEmbeddings(model_name=owngptsettings.embeddings_model_name)
    chroma_client = chromadb.PersistentClient(path=owngptsettings.persist_directory,settings=Settings(anonymized_telemetry=False))
    db = Chroma(persist_directory=owngptsettings.persist_directory,collection_name=owngptsettings.collection_name, embedding_function=embeddings, client = chroma_client)
    retriever = db.as_retriever()
    
    # Prepare the LLM
    match owngptsettings.model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=owngptsettings.model_path, n_ctx=owngptsettings.model_n_ctx, n_threads = 8, verbose=verbose_output)
        case "GPT4All":
            llm = GPT4All(model=owngptsettings.model_path, n_ctx=owngptsettings.model_n_ctx, backend='gptj', verbose=verbose_output)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    # Get the answer from the chain
    res = qa(human_msg)
    #  print(res)   
    answer, docs = res['result'], res['source_documents']
    return answer, docs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
        response,docs = private_gpt_generate_msg(user_input,True)
        print('Summary output ===========')
        print(f'QUESTION: {user_input}.')
        print(f'RESPONSE: {response}.')
        print('Reference data ==========')
        print(docs)
    else:
        print("Please provide a command line question")
 
        
