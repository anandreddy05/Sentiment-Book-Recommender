import pandas as pd
import numpy as np
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import ChatHuggingFace
import os
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import gradio as gr

# Load environment variables
load_dotenv()

# Load book dataset
books = pd.read_csv("./datasets/books_with_image_paths.csv")
books["images"] = books["Img_Path"].astype(str)
books["images"] = np.where(books["images"].isna(), "CoverNotFound.jpg", books["images"])

# Load tagged descriptions
raw_docs = TextLoader("./notebooks/tagged_desc.txt", encoding="utf-8").load()

# Split text into smaller chunks
text_splitters = CharacterTextSplitter(chunk_size=4000, chunk_overlap=200, separator="\n\n")
documents = text_splitters.split_documents(raw_docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cuda'})

# Create FAISS vector store
db_books = FAISS.from_documents(documents=documents, embedding=embeddings)

# Function to retrieve similar books
def retrieve_similar_books(query: str,
                           category: str = None,
                           sentiment: str = None,
                           initial_top_k=50,
                           final_top_k=10) -> pd.DataFrame:
    recs = db_books.similarity_search(query=query, k=initial_top_k)
    books_list = []

    for rec in recs:
        match = re.search(r'BK\d{6}', rec.page_content)
        if match:
            books_list.append(match.group())

    print(f"🔹 Total Extracted Book IDs: {len(books_list)}")

    # Check if extracted books exist in DataFrame
    missing_books = [bid for bid in books_list if bid not in books["FormattedBookID"].values]
    print(f"❌ Missing Books (Not Found in Dataset): {len(missing_books)} -> {missing_books}")

    books_recs = books[books["FormattedBookID"].isin(books_list)]
    print(f"✅ Books Matched in DataFrame: {len(books_recs)}")

    # Apply category filter
    if category != "All":
        print(f"🔹 Before Category Filter: {len(books_recs)}")
        books_recs = books_recs[books_recs["Mapped_Genre"] == category]
        print(f"✅ After Category Filter: {len(books_recs)}")

    # Apply sentiment filter
    sentiment_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Sad": "sadness",
        "Angry": "anger",
        "Fear": "fear"
    }
    
    if sentiment != "None" and sentiment in sentiment_map:
        target_sentiment = sentiment_map[sentiment]
        print(f"🔹 Before Sentiment Filter: {len(books_recs)}")
        books_recs = books_recs[books_recs["sentiment"] == target_sentiment]
        print(f"✅ After Sentiment Filter: {len(books_recs)}")

    # Apply final top_k limit
    books_recs = books_recs.head(final_top_k)
    print(f"📌 Final Books to Show: {len(books_recs)}")

    return books_recs


# Function to generate recommendations
def recommend_books(query: str, category: str, sentiment: str):
    recommendations = retrieve_similar_books(query, category, sentiment, initial_top_k=50, final_top_k=15)
    result = []

    for _, row in recommendations.iterrows():
        # print(f"Image URL: {row['images']}")  
        # print(f"Author: {row['Author']} - Description: {row['Description'][:50]}...")  # Debugging

        description = row["Description"]
        limit_desc_split = description.split()
        limit_desc = " ".join(limit_desc_split[:30]) + "..."

        author = row["Author"]
        caption = f"By {author}: {limit_desc}"

        result.append((row["images"], caption))

    return result

# Create Gradio interface
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo").set(
    body_background_fill="*neutral_50",
    block_background_fill="*neutral_100",
    button_primary_background_fill="*primary_500",
)

categories = ["All", "Fiction", "Non-Fiction"]
sentiments = ["None", "Happy", "Surprising", "Sad", "Angry", "Fear"]

with gr.Blocks(theme=theme, title="Book Recommendation System") as app:
    gr.Markdown("# 📚 Smart Book Recommendations\nFind your next great read based on your interests and mood!")

    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
            label="What kind of book are you looking for?",
            placeholder="E.g., 'A thrilling mystery set in Victorian London'",
            lines=2  
            )
            with gr.Row():
                category_dropdown = gr.Dropdown(choices=categories, label="Book Category", value="All")
                sentiment_dropdown = gr.Dropdown(choices=sentiments, label="Mood/Sentiment", value="None")
            
            submit_btn = gr.Button("Find Books", variant="primary")

    with gr.Row():
        output_gallery = gr.Gallery(label="Recommended Books",
                                    show_label=True, 
                                    elem_id="gallery",
                                    columns=[4], rows=[7],
                                    height="auto", 
                                    allow_preview=True)

    submit_btn.click(fn=recommend_books, inputs=[query_input, category_dropdown, sentiment_dropdown], outputs=output_gallery)

    gr.Markdown(
        """
        ### 📖 How it works  
        - **Query**: Describe what you're looking for in natural language  
        - **Category**: Filter by book genre  
        - **Mood**: Find books that match your desired emotional experience  
        """
    )

if __name__ == "__main__":
    app.launch(debug=True)