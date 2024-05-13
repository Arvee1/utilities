import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import speech_recognition as sr
import replicate
import pyaudio
import wave
from audiorecorder import audiorecorder
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Replicate
from langchain.chains import ConversationChain
from langchain_core.prompts.prompt import PromptTemplate

# initialize
r = sr.Recognizer()
# This is in seconds, this will control the end time of the record after the last sound was made
r.pause_threshold = 2

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ofsc_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
     model_name=EMBED_MODEL
 )

collection = client.get_or_create_collection(
     name=COLLECTION_NAME,
     embedding_function=embedding_func,
     metadata={"hnsw:space": "cosine"},
 )

# Load VectorDB
# if st.sidebar.button("Load OFSC Facsheets into Vector DB if loading the page for the first time.", type="primary"):
@st.cache_resource
def create_vector():
      with open("Renting-Book-October-2023-Update.txt") as f:
          hansard = f.read()
          text_splitter = RecursiveCharacterTextSplitter(
              chunk_size=500,
              chunk_overlap=20,
              length_function=len,
              is_separator_regex=False,
          )
           
      texts = text_splitter.create_documents([hansard])
      documents = text_splitter.split_text(hansard)[:len(texts)]
     
      collection.add(
           documents=documents,
           ids=[f"id{i}" for i in range(len(documents))],
      )
      f.close()

create_vector()

llm = Replicate(
    model="meta/meta-llama-3-8b-instruct",
    model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
)

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

conversation_buf = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
    # memory=ConversationBufferMemory(),
)

# The UI Part
st.title("👨‍💻 Wazzup!!!! What do you want to know about Renting in Canberra?")
prompt = st.text_area("Please enter what you want to know about renting in Canberra.")

if st.button("Submit to AI", type="primary"):
     # query_results = collection.query(
          # query_texts=[prompt],
          # include=["documents", "embeddings"],
          # where_document={"$contains":prompt},
          # include=["documents"],
          # n_results=20,
     # )
     # augment_query = str(query_results["documents"])
     # augment_input = "Prompt: " + prompt + " " + augment_query
     # st.write(augment_input)
     ai_response = conversation_buf.run(prompt)
     st.write(ai_response)
     # st.write(conversation_buf.memory.buffer)

     # result_ai = ""
     # The meta/meta-llama-3-70b-instruct model can stream output as it's running.
     # for event in replicate.stream(
         # "meta/meta-llama-3-70b-instruct",
         # input={
             # "top_k": 50,
             # "top_p": 0.9,
             # "prompt": "Prompt: " + prompt + " " + augment_query,
             # "max_tokens": 512,
             # "min_tokens": 0,
             # "temperature": 0.6,
             # "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
             # "presence_penalty": 1.15,
             # "frequency_penalty": 0.2
         # },
     # ):
         # result_ai = result_ai + (str(event))
     # st.write(result_ai)
     

# This is the part where you can verbally ask about stuff
audio = audiorecorder("Click to record", "Click to stop recording")
      
