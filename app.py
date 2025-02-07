import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import ChatHuggingFace
import os
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
books = pd.read_csv("./datasets/books_with_image_paths.csv")

books["images"] = books["Img_Path"] + "&fife=w800"

books["images"] = np.where(books["images"].isna(),
                           "CoverNotFound.jpg",
                           books["images"])

raw_docs = TextLoader("./notebooks/tagged_desc.txt",encoding="utf-8").load()
text_splitters = CharacterTextSplitter(chunk_size=1500,chunk_overlap=0,separator="\n")
documents = text_splitters.split_documents(raw_docs)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs = {'device':'cuda'}
)
db_books = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

def retrieve_similar_books(
    query:str,
    category:str = None,
    sentiment: str = None,
    initial_top_k = 50,
    final_top_k = 10,
) ->pd.DataFrame:
    recs = db_books.similarity_search(query=query,k = initial_top_k)
    