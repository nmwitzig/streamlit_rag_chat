#%%
import pandas as pd
import streamlit as st
from io import StringIO
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from helper_utils import word_wrap
import torch
# fixes weird torch error https://discuss.streamlit.io/t/message-error-about-torch/90886/5
torch.classes.__path__ = [] 

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()

import os
import openai
from openai import OpenAI

# Setting the API key

XAI_API_KEY = os.environ['GROK_API_KEY']
openai_client = OpenAI(
  api_key=XAI_API_KEY,
  base_url="https://api.groq.com/openai/v1",
)



def rag(query, retrieved_documents, model="llama-3.1-8b-instant"):
    information = "\n\n".join(retrieved_documents)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert research assistant. Your users are asking questions about information contained in an uploaded document."
            "You will be shown the user's question, and the relevant information from the document. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

# markdown
st.markdown("## Chat with your file")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
if uploaded_file is not None:
    
    reader = PdfReader(uploaded_file)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]
    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]
    
    
    # streamlit show green bar documentat uploaded
    st.success('Document uploaded successfully!')

    

    # show the text
    #st.write(pdf_texts[0])
    with st.spinner('Processing text... (Splitting, Sentence Tokenizing, Embedding)'): 
        character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=0)
        character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

        #st.write(f"\nTotal chunks: {len(character_split_texts)}")


        token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

        token_split_texts = []
        for text in character_split_texts:
            token_split_texts += token_splitter.split_text(text)

        #st.write(word_wrap(token_split_texts[10]))
        #st.write(f"\nTotal chunks: {len(token_split_texts)}")

        #st.write(embedding_function([token_split_texts[10]]))

        chroma_collection = chroma_client.get_or_create_collection(uploaded_file.name.removesuffix('.pdf'), embedding_function=embedding_function)

        ids = [str(i) for i in range(len(token_split_texts))]

        chroma_collection.add(ids=ids, documents=token_split_texts)
        chroma_collection.count()
        
        st.write("Done! You can now write your query!")

        # prompt = st.chat_input("You can ask me anything about the document! 👋")
        # if prompt:
        #     st.write(f"User has sent the following prompt: {prompt}")


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Write your query here 👋"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})




        results = chroma_collection.query(query_texts=[prompt], n_results=5)
        retrieved_documents = results['documents'][0]

        #for document in retrieved_documents:
        #    st.write(word_wrap(document))
        #    st.write('\n')

        output = rag(query=prompt, retrieved_documents=retrieved_documents)
        response = f"Echo: {word_wrap(output)}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


    


    
    #st.status("Done!")
    
