# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:29:30 2024

@author: yash
"""

import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
groq_api_key=os.getenv("GROQ_API_KEY")
## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


llm=ChatGroq(groq_api_key=groq_api_key,model_name='llama3-70b-8192')


st.set_page_config(page_title="LANGCHAIN: Summarize Text ", page_icon="🦜")
st.title("LANGCHAIN: Summarize Text ")
st.subheader('Summarize URL')


uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
documents=[]
        ## Process uploaded  PDF's
if uploaded_files:
    
    
    
    for uploaded_file in uploaded_files:
        
        temppdf=f"./temp.pdf"
        
        with open(temppdf,"wb") as file:
            
            file.write(uploaded_file.getvalue())
            
            file_name=uploaded_file.name
            

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
#st.write(documents)


final_documents=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100).split_documents(documents)
final_documents
st.write(len(final_documents))


chunks_prompt="""
Please summarize the below speech:
Speech:`{text}'
Summary:
"""
map_prompt_template=PromptTemplate(input_variables=['text'],
                                    template=chunks_prompt)    


final_prompt='''
Provide the final summary of the entire pdf with these important points.
provide the summary in number 
points for the speech.
Speech:{text}

'''
final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)
final_prompt_template   

summary_chain=load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt_template,
    combine_prompt=final_prompt_template,
    verbose=True
)

output=summary_chain.run(final_documents)
st.write(output)     
            
            