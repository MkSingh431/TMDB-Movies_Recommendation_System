import streamlit as st
import pickle
import pandas as pd
import requests
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Function to Fetch Posters ---
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=0737e3e83144018bce73b00411bc39bd&language=en-US".format(movie_id)
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get('poster_path')
        if not poster_path:
            return "https://via.placeholder.com/500x750?text=No+Image"
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception:
        return "https://via.placeholder.com/500x750?text=Error"

# --- Recommendation Logic ---
def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_movies = []
        recommended_posters = []

        for i in movies_list:
            movie_row = movies.iloc[i[0]]
            recommended_movies.append(movie_row.title)
            
            # Handle ID columns safely
            if 'movie_id' in movies.columns:
                recommended_posters.append(fetch_poster(movie_row.movie_id))
            elif 'id' in movies.columns:
                recommended_posters.append(fetch_poster(movie_row.id))
            else:
                recommended_posters.append("https://via.placeholder.com/500x750?text=No+ID")

        return recommended_movies, recommended_posters
    except Exception as e:
        st.error(f"Error generating recommendation: {e}")
        return [], []

# --- Main App Execution ---
st.title('🎬 Movie Recommender System')

# Define file paths
pkl_file = 'model/movie_list.pkl'
sim_file = 'model/similarity.pkl'
csv_file = 'tmdb_5000_movies.csv'

movies = None
similarity = None

# --- 1. Try Loading Pre-built Pickle Files (Safely) ---
# We wrap this in a TRY block so if the file is "fake" (LFS pointer), it fails gracefully.
if os.path.exists(pkl_file) and os.path.exists(sim_file):
    try:
        with open(pkl_file, 'rb') as f:
            movies = pickle.load(f)
        with open(sim_file, 'rb') as f:
            similarity = pickle.load(f)
        # Check if loaded data is valid (not None)
        if movies is None or similarity is None:
            raise ValueError("Loaded pickle data is empty")
            
    except (pickle.UnpicklingError, EOFError, ValueError, Exception) as e:
        # This catches the "\x0a" error!
        st.warning(f"Pickle files found but corrupted (likely Git LFS pointers). Switching to CSV fallback mode...")
        movies = None
        similarity = None

# --- 2. Fallback: Build Model from CSV ---
if movies is None or similarity is None:
    if os.path.exists(csv_file):
        with st.spinner('Building model from CSV... This may take a moment.'):
            try:
                # Load Data
                df = pd.read_csv(csv_file)
                
                # Preprocessing
                df['overview'] = df['overview'].fillna('')
                # Ensure title column exists
                if 'original_title' in df.columns and 'title' not in df.columns:
                    df['title'] = df['original_title']
                
                # Create tags for similarity
                df['tags'] = df['overview'] 
                
                # Vectorize (Convert text to numbers)
                cv = CountVectorizer(max_features=5000, stop_words='english')
                vectors = cv.fit_transform(df['tags'].values.astype('U'))
                
                # Calculate Cosine Similarity
                similarity = cosine_similarity(vectors)
                movies = df
                
                st.success("Model built successfully from CSV!")
            except Exception as e:
                st.error(f"Critical Error building model from CSV: {e}")
                st.stop()
    else:
        st.error("❌ File Not Found: Please upload 'tmdb_5000_movies.csv' to your GitHub repository.")
        st.stop()

# --- UI Layout ---
if movies is not None:
    selected_movie_name = st.selectbox(
        'Type or select a movie from the dropdown',
        movies['title'].values
    )

    if st.button('Recommend'):
        names, posters = recommend(selected_movie_name)
        
        if names:
            col1, col2, col3, col4, col5 = st.columns(5)
            cols = [col1, col2, col3, col4, col5]
            for idx, col in enumerate(cols):
                with col:
                    st.text(names[idx])
                    st.image(posters[idx])
