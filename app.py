import streamlit as st
import pandas as pd
import numpy as np
import webbrowser
from dotenv import load_dotenv

import faiss
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Load book data for Streamlit app
books = pd.read_csv("./datasets/books_with_image_paths.csv")

# Load and process book descriptions
raw_documents = TextLoader("./notebooks/tagged_desc.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

# Load FAISS embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
db_books = FAISS.from_documents(documents=documents, embedding=embeddings)

# Streamlit app setup
st.set_page_config(page_title="Sentimental Book Recommender", layout="wide")

st.title("📚 Sentimental Book Recommender")


# Function to retrieve recommendations
def retrieve_semantics_recommendations(query: str, genre: str = None, emotion: str = None, initial_top_k: int = 50, final_top_k: int = 16):
    book_list = []

    # Search for similar descriptions using FAISS
    recs = db_books.similarity_search(query, k=initial_top_k)
    
    # Extract Book IDs from the results
    for rec in recs:
        first_word = rec.page_content.strip().split()[0]
        if first_word.startswith("BK") and first_word[2:].isdigit():
            book_list.append(first_word)

    # Filter books based on Book_IDs found in FAISS search
    book_recs = books[books["BookID"].isin(book_list)].head(initial_top_k)

    # Apply Genre filter
    if genre and genre != "All":
        book_recs = book_recs[book_recs["Filtered_Genres"].astype(str).str.contains(genre, case=False, na=False)]

    # Apply Emotion filter
    if emotion and emotion != "All":
        book_recs = book_recs[book_recs["sentiment"] == emotion]

    return book_recs.head(final_top_k)  # Return top 16 recommendations


# Sidebar user input
st.sidebar.header("🔍 Find a Book")

user_query = st.sidebar.text_input("Enter a book description:", placeholder="e.g., A story about adventure and discovery")

# Genre selection from unique genres in the dataset
genre = st.sidebar.selectbox("Select a Genre:", ["All"] + sorted(books["Filtered_Genres"].dropna().unique()))

# Emotion selection from unique emotions in the dataset
emotion = st.sidebar.selectbox("Select an Emotion:", ["All", "Happy", "Suprising", "Angry", "Suspenseful", "Sad"])

if st.sidebar.button("Find Recommendations"):
    with st.spinner("Fetching recommendations..."):
        recommendations = retrieve_semantics_recommendations(user_query, genre, emotion)

        if recommendations.empty:
            st.warning("No books found. Try a different query or filters.")
        else:
            st.subheader("📖 Recommended Books:")
            cols_per_row = 4  # Number of books per row
            cols = st.columns(cols_per_row)

            for i, (_, row) in enumerate(recommendations.iterrows()):
                with cols[i % cols_per_row]:  # Distribute books across columns
                    st.image(row["Img_Path"], width=150, caption=row["Book"])
                    st.write(f"**{row['Book']}** by {row['Author']}")
                    st.write(row["Description_x"][:150] + "...")  # Truncate description
                    
                    # Button to open book URL
                    if st.button(f"📖 Read More", key=f"book_{row['BookID']}"):
                        webbrowser.open(row["URL"])  # Ensure "URL" exists in your CSV

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Tip:** Try different descriptions and emotions for better results!")
