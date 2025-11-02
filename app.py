# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from rapidfuzz import fuzz, process
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Text preprocessing
@st.cache_data
def preprocess_text(text):
    """Clean and normalize text for better matching."""
    if pd.isna(text):
        return ""
    # Convert to string and lowercase
    text = str(text).lower()
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load and clean data
@st.cache_data
def load_and_prepare_data():
    try:
        logging.info("Loading and preparing data...")
        df = pd.read_csv("books.csv", on_bad_lines='skip', low_memory=False)

        if 'Unnamed: 12' in df.columns:
            df = df.drop(columns=['Unnamed: 12'])

        for col in ['average_rating', 'num_pages', 'ratings_count']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Preprocess text columns for search
        if 'title' in df.columns:
            df['title_processed'] = df['title'].apply(preprocess_text)
        if 'authors' in df.columns:
            df['authors_processed'] = df['authors'].apply(preprocess_text)
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

    if 'language_code' in df.columns:
        df['language_code'] = df['language_code'].fillna('unknown')

    if 'title' in df.columns:
        df['title'] = df['title'].astype(str).fillna('Unknown Title')

    df = df.drop_duplicates().reset_index(drop=True)
    required = [c for c in ['average_rating', 'ratings_count', 'num_pages'] if c in df.columns]
    df_clean = df.dropna(subset=required).copy().reset_index(drop=True)

    return df_clean

# Train model
@st.cache_resource
def train_model(df_clean):
    df_model = df_clean.reset_index(drop=True).copy()
    df_model['rating_between'] = pd.cut(df_model['average_rating'],
                                   bins=[-0.01, 1, 2, 3, 4, 5],
                                   labels=["0-1", "1-2", "2-3", "3-4", "4-5"])
    if 'language_code' not in df_model.columns:
        df_model['language_code'] = 'unknown'
    else:
        df_model['language_code'] = df_model['language_code'].fillna('unknown')

    rating_df = pd.get_dummies(df_model['rating_between'])
    language_df = pd.get_dummies(df_model['language_code'])

    numerical = df_model[['average_rating', 'ratings_count']].copy()
    feature_df = pd.concat([rating_df, language_df, numerical], axis=1).fillna(0)

    scaler = MinMaxScaler()
    features = scaler.fit_transform(feature_df)

    model = neighbors.NearestNeighbors(n_neighbors=8, algorithm='ball_tree', metric='euclidean')
    model.fit(features)
    distances, indices = model.kneighbors(features)

    return df_model, indices

# Recommender
def recommend_books(name, df_model, indices):
    matches = df_model[df_model['title'].str.lower() == name.lower()]
    if matches.empty:
        possible = difflib.get_close_matches(name, df_model['title'].tolist(), n=3, cutoff=0.5)
        if not possible:
            return {"error": "Book not found and no close match", "suggestions": []}
        best = possible[0]
        book_idx = df_model[df_model['title'] == best].index[0]
        suggestion_used = best
    else:
        book_idx = matches.index[0]
        suggestion_used = df_model.loc[book_idx, 'title']

    recs = []
    for nb in indices[book_idx]:
        if nb == book_idx:
            continue
        recs.append({
            "title": df_model.loc[nb, 'title'],
            "authors": df_model.loc[nb, 'authors'] if 'authors' in df_model.columns else "Unknown",
            "average_rating": df_model.loc[nb, 'average_rating'],
            "ratings_count": int(df_model.loc[nb, 'ratings_count']) if not pd.isna(df_model.loc[nb, 'ratings_count']) else "N/A"
        })

    return {"query_used": suggestion_used, "recommendations": recs}

# App UI
st.set_page_config(layout="wide")
st.title("📚 Book Recommendation System")

# Load and train
df_clean = load_and_prepare_data()
df_model, indices = train_model(df_clean)

# Sidebar
st.sidebar.header("🔎 Search Options")
search_option = st.sidebar.radio("Search by:", ("Book Title", "Author"))

# Book search
# if search_option == "Book Title":
#     book_name = st.sidebar.text_input("Enter book title:")
#     if book_name:
#         st.subheader(f"🔍 Search results for: {book_name}")
#         results = df_model[df_model['title'].str.lower().str.contains(book_name.lower())]
#         if not results.empty:
#             st.dataframe(results[['title', 'authors', 'average_rating', 'ratings_count']].head(10))
#         else:
#             st.warning("No matching books found.")

# Advanced search function
@st.cache_data
def advanced_search(query, df, column='title_processed', min_score=40):  # Even lower threshold
    """
    Perform advanced search using TF-IDF and fuzzy matching.
    Returns DataFrame of results sorted by relevance.
    """
    try:
        logging.info(f"Performing advanced search for query: {query}")
        processed_query = preprocess_text(query)
        
        # Try fuzzy matching first for short queries
        if len(processed_query.split()) <= 2:  # For short queries like "hary"
            matches = process.extract(
                processed_query,
                df[column].fillna(''),
                scorer=fuzz.ratio,  # Using simple ratio for better partial matches
                limit=10
            )
            
            fuzzy_indices = []
            scores = []
            
            # Get the actual indices from the DataFrame
            for match in matches:
                text = match[0]  # The matched text
                score = match[1]  # The match score
                idx = df[df[column] == text].index[0]  # Get the actual DataFrame index
                
                # Boost score for prefix matches (e.g., "hary" matching "harry")
                if text.startswith(processed_query) or processed_query.startswith(text):
                    score += 15
                if score >= min_score:
                    fuzzy_indices.append(idx)
                    scores.append(min(score, 100))  # Cap at 100
            
            if fuzzy_indices:
                results_df = df.iloc[fuzzy_indices].copy()
                results_df['match_score'] = scores
                return results_df.sort_values('match_score', ascending=False)
        
        # If fuzzy matching didn't work or for longer queries, try TF-IDF
        tfidf = TfidfVectorizer(min_df=1, stop_words='english', 
                               ngram_range=(1, 3))  # Added trigrams for better matching
        
        # Get all texts and fit TF-IDF
        texts = df[column].fillna('').tolist()
        tfidf_matrix = tfidf.fit_transform(texts)
        
        # Transform query and calculate similarities
        query_vec = tfidf.transform([processed_query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get indices of results above similarity threshold
        tfidf_indices = np.where(similarities > 0.01)[0]  # Even lower similarity threshold
        
        if len(tfidf_indices) == 0:
            # If no TF-IDF matches, try fuzzy matching
            matches = process.extract(
                processed_query,
                df[column].fillna(''),
                scorer=fuzz.ratio,
                limit=10
            )
            
            # Filter matches above score threshold
            fuzzy_indices = []
            scores = []
            
            # Process matches and get actual DataFrame indices
            for match in matches:
                text = match[0]  # The matched text
                score = match[1]  # The match score
                if score >= min_score:
                    idx = df[df[column] == text].index[0]  # Get the actual DataFrame index
                    fuzzy_indices.append(idx)
                    scores.append(score)
            
            if fuzzy_indices:
                results_df = df.iloc[fuzzy_indices].copy()
                results_df['match_score'] = scores
            else:
                return pd.DataFrame()
        else:
            # Use TF-IDF results
            results_df = df.iloc[tfidf_indices].copy()
            results_df['match_score'] = similarities[tfidf_indices] * 100
        
        # Sort by match score
        results_df = results_df.sort_values('match_score', ascending=False)
        return results_df
    
    except Exception as e:
        logging.error(f"Error in advanced_search: {str(e)}")
        return pd.DataFrame()

# Search interface
if search_option == "Book Title":
    search_query = st.sidebar.text_input("Enter book title:", key="book_search_input")
    search_column = 'title_processed'
elif search_option == "Author":
    search_query = st.sidebar.text_input("Enter author name:", key="author_search_input")
    search_column = 'authors_processed'

if search_query:
    try:
        st.subheader(f"🔍 Search results for: {search_query}")
        
        # Perform advanced search with lower threshold for better matching
        results = advanced_search(search_query, df_model, column=search_column, min_score=45)
        
        if not results.empty:
            # Display results with match score
            display_cols = ['title', 'authors', 'average_rating', 'ratings_count', 'match_score']
            display_df = results[display_cols].head(10)
            
            # Format match score and sort results
            display_df['match_score'] = display_df['match_score'].apply(lambda x: f"{float(x):.1f}%")
            if search_option == "Author":
                display_df = display_df.sort_values(['match_score', 'average_rating'], ascending=[False, False])
            
            st.dataframe(display_df)
            
            # Show suggestion if exact match wasn't found
            first_score = float(str(results.iloc[0]['match_score']).rstrip('%'))
            if first_score < 100.0:
                st.info(f"📚 Showing similar results. Did you mean: {results.iloc[0]['title' if search_option == 'Book Title' else 'authors']}?")
            
            # Log successful search
            logging.info(f"Found {len(results)} results for {search_option.lower()}: {search_query}")
        else:
            st.warning(f"No matching {search_option.lower()}s found.")
            logging.info(f"No results found for {search_option.lower()}: {search_query}")
            
    except Exception as e:
        st.error("An error occurred while searching. Please try again.")
        logging.error(f"Search error for {search_option.lower()} '{search_query}': {str(e)}")

# Recommendation
st.markdown("---")
st.subheader("🎯 Book Recommender")
user_input = st.text_input("Enter a book name to get recommendations:")
if user_input:
    output = recommend_books(user_input, df_model, indices)
    if "error" in output:
        st.error(output["error"])
    else:
        # st.success(f"Recommendations based on: {output['query_used']}")
        st.table(pd.DataFrame(output["recommendations"]))

# Top 10 books
st.markdown("---")
st.subheader("🏆 Top 10 Books by Ratings Count")
top_books = df_clean.sort_values("ratings_count", ascending=False).head(10)
st.table(top_books[['title', 'authors', 'average_rating', 'ratings_count']])

# Top 10 authors
st.subheader("🖋️ Top 10 Authors with Most Books")

# Group and sort authors by book count
top_authors = df_clean.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False).head(10)
top_authors = top_authors.rename(columns={"title": "book_count"}).reset_index(drop=True)

# Loop through top authors and make them expandable
for i, row in top_authors.iterrows():
    with st.expander(f"📖 {row['authors']} — {row['book_count']} books"):
        author_books = df_clean[df_clean['authors'] == row['authors']][['title', 'average_rating', 'ratings_count']]
        st.dataframe(author_books.sort_values('average_rating', ascending=False).reset_index(drop=True))

# # Optional plots
# Separator and heading
st.markdown("---")
st.markdown("## 📊 View Distribution and Relationship Plots")

# Checkbox to toggle visibility
show_plots = st.checkbox("📌 Show statistical visualizations", value=False)

if show_plots:
    # 1. Distribution Plot
    st.markdown("### ⭐ Distribution of Average Ratings")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(df_clean['average_rating'], kde=True, ax=ax1)
    ax1.set_title('Distribution of Average Ratings')
    ax1.set_xlabel('Average Rating')
    st.pyplot(fig1)

    st.markdown("""
    📌 **Explanation:**  
    This histogram shows how books are rated on average. Most books tend to fall between **3 and 4.5 stars**, with a sharp peak around 4.  
    This means readers generally give favorable ratings to books.
    """)

    # 2. Rating vs Ratings Count
    st.markdown("### 🔁 Average Rating vs. Ratings Count (Log Scale)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_clean.sample(min(2000, len(df_clean))),
                    x='average_rating', y='ratings_count', alpha=0.6, ax=ax2)
    ax2.set_yscale('log')
    ax2.set_title('Average Rating vs Ratings Count')
    st.pyplot(fig2)

    st.markdown("""
    📌 **Explanation:**  
    This scatter plot shows whether **popular books** (those with more ratings) tend to have **higher or lower average ratings**.  
    A log scale is used because ratings count varies widely.  
    👉 Most highly rated books also have high ratings count — indicating **popular books are often well-liked**.
    """)

    # 3. Rating vs Number of Pages
    st.markdown("### 📘 Average Rating vs. Number of Pages")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_clean.sample(min(2000, len(df_clean))),
                    x='average_rating', y='num_pages', alpha=0.6, ax=ax3)
    ax3.set_title('Average Rating vs Number of Pages')
    st.pyplot(fig3)

    st.markdown("""
    📌 **Explanation:**  
    This plot explores whether **longer books get better ratings**.  
    There’s **no strong trend**, but some longer books (> 500 pages) still have **very high ratings**, showing quality matters more than length.
    """)

    # Optional: Info box
    with st.expander("ℹ️ Why are these plots useful?"):
        st.markdown("""
        - Understand **user rating behavior**.
        - See which types of books are most liked or rated.
        - Spot trends in book length, popularity, and ratings.
        """)
