import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os

def load_movie_data(csv_path):
    return pd.read_csv(csv_path)

def get_embeddings():
    embedding_model_name = "text-embedding-ada-002"
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    return OpenAIEmbeddings(model=embedding_model_name, openai_api_key=openai_api_key)

def get_chat_model():
    chat_model_name = st.secrets["OPENAI_Model"]
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    return ChatOpenAI(openai_api_key=openai_api_key, model=chat_model_name)

def create_new_vector_store(_embeddings, movie_data, faiss_index_path):
    movie_data['combined_text'] = movie_data.apply(lambda row: f"Movie: {row['names']}\nGenre: {row['genre']}\nRating: {row['score']}\nOverview: {row['overview']}", axis=1)
    texts = movie_data['combined_text'].tolist()
    metadatas = movie_data.to_dict('records')
    vector_store = FAISS.from_texts(texts, _embeddings, metadatas=metadatas)
    vector_store.save_local(faiss_index_path)
    st.success("Created and saved new FAISS index.")
    return vector_store

@st.cache_resource
def load_or_create_vector_store(_embeddings, movie_data):
    faiss_index_path = 'faiss_index'
    if os.path.exists(faiss_index_path):
        try:
            vector_store = FAISS.load_local(faiss_index_path, _embeddings, allow_dangerous_deserialization=True)
            st.success("Loaded existing FAISS index.")
        except Exception as e:
            st.warning(f"Error loading existing index: {e}. Creating a new one.")
            vector_store = create_new_vector_store(_embeddings, movie_data, faiss_index_path)
    else:
        vector_store = create_new_vector_store(_embeddings, movie_data, faiss_index_path)
    return vector_store

st.title("Movie Recommendation Chatbot")
st.write("Chat with me about your movie preferences, and I'll recommend some films!")

if 'initialized' not in st.session_state:
    with st.spinner("Initializing chatbot..."):
        csv_path = './data/imdb_movies.csv'
        movie_data = load_movie_data(csv_path)
        embeddings = get_embeddings()
        vector_store = load_or_create_vector_store(embeddings, movie_data)
        chat_model = get_chat_model()
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        prompt_template = """
        You are a friendly movie recommendation chatbot. Use the following pieces of context to answer the human's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        When recommending movies or answering questions about movies, always include the movie name, genre, rating (out of 100), and a brief summary of the overview.

        {context}

        Human: {question}
        Chatbot:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=chat_model,
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        
        st.session_state.conversation_chain = conversation_chain
        st.session_state.chat_history = []
        st.session_state.initialized = True

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("What kind of movie are you in the mood for?")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Thinking..."):
        response = st.session_state.conversation_chain({"question": user_input})
        bot_response = response['answer']
    
    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})