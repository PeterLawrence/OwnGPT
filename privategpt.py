# -*- coding: utf-8 -*-
"""
Create ownChat test private gpt
@author: Lawrence P.J
"""

import chromadb
from chromadb.config import Settings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import GPT4All, LlamaCpp

import owngptsettings
import sys

# next two lines for fixing a warning in torch
import torch
torch.classes.__path__ = []

def private_gpt_generate_msg(human_msg,verbose_output):
    global res
    embeddings = HuggingFaceEmbeddings(model_name=owngptsettings.embeddings_model_name)
    chroma_client = chromadb.PersistentClient(path=owngptsettings.persist_directory,settings=Settings(anonymized_telemetry=False))
    db = Chroma(persist_directory=owngptsettings.persist_directory,collection_name=owngptsettings.collection_name, embedding_function=embeddings, client = chroma_client)
    retriever = db.as_retriever()
    
    # Prepare the LLM
    match owngptsettings.model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=owngptsettings.model_path, n_ctx=owngptsettings.model_n_ctx, n_threads = 8, n_batch=512, verbose=verbose_output)
        case "GPT4All":
            llm = GPT4All(model=owngptsettings.model_path, n_ctx=owngptsettings.model_n_ctx, backend='gptj', verbose=verbose_output)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;

    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Context: {context}"
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            ]
        )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # note the next statement populates {context} automatically in system_prompt above
    # this is the documents information
    chain = create_retrieval_chain(retriever, question_answer_chain)

    # Get the answer from the chain
    res = chain.invoke({"input": human_msg})
    if 'answer' in res:
        answer = res['answer']
    else:
        answer = 'Response error'
    docs = []
    if 'context' in res:
        for entry in res['context']:
            if 'source' in entry.metadata:
                if not(entry.metadata['source'] in docs):
                    docs.append(entry.metadata['source'])
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
 
        
