import os
import pickle
import streamlit as st
import requests
import pandas as pd
import numpy as np
import traceback

# Define paths
pkl_path = 'model/movie_list.pkl'
sim_path = 'model/similarity.pkl'
csv_path = 'tmdb_5000_movies.csv'
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=0737e3e83144018bce73b00411bc39bd&language=en-US".format(movie_id)
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get('poster_path')
        if not poster_path:
            # fallback placeholder when poster missing
            return "https://via.placeholder.com/500x750?text=No+Image"
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    except Exception:
        return "https://via.placeholder.com/500x750?text=No+Image"

def recommend(movie):
    if movies is None or similarity is None:
        return [], []
    try:
        idxs = movies[movies['title'] == movie].index
        if len(idxs) == 0:
            st.error(f'Movie "{movie}" not found in dataset.')
            return [], []
        index = idxs[0]
    except Exception:
        st.error('Error finding selected movie in dataset.')
        return [], []

    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster (support either 'movie_id' or 'id' column)
        row = movies.iloc[i[0]]
        if 'movie_id' in movies.columns:
            movie_id = row.movie_id
        elif 'id' in movies.columns:
            movie_id = row.id
        else:
            movie_id = None
        recommended_movie_posters.append(fetch_poster(movie_id) if movie_id else "https://via.placeholder.com/500x750?text=No+Image")
        recommended_movie_names.append(row.title)

    return recommended_movie_names,recommended_movie_posters


try:
    st.header('Movie Recommender System')

    # Load models with graceful fallbacks
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    movie_list_path = os.path.join(model_dir, 'movie_list.pkl')
    similarity_path = os.path.join(model_dir, 'similarity.pkl')

    movies = None
    similarity = None

    if os.path.exists(movie_list_path):
        try:
            with open(movie_list_path, 'rb') as f:
                movies = pickle.load(f)
        except Exception as e:
            st.warning(f'Could not load movie_list.pkl: {e}')

    if os.path.exists(similarity_path):
        try:
            with open(similarity_path, 'rb') as f:
                similarity = pickle.load(f)
        except Exception as e:
            st.warning(f'Could not load similarity.pkl: {e}')

    # Fallback: try loading raw CSV if pickle models are missing
    if movies is None:
        csv_path = os.path.join(os.path.dirname(__file__), 'tmdb_5000_movies.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # normalize column names
                if 'title' in df.columns:
                    movies = df[['title']].copy()
                elif 'original_title' in df.columns:
                    movies = df[['original_title']].rename(columns={'original_title': 'title'}).copy()
                else:
                    movies = pd.DataFrame({'title': df.iloc[:,0]})
                # include id column if present
                if 'id' in df.columns:
                    movies['id'] = df['id']
            except Exception as e:
                st.error(f'Failed to read fallback CSV: {e}')

    # If still no movies, show error and stop
    if movies is None:
        st.error('Model files not found and fallback CSV not available. Place `movie_list.pkl` and `similarity.pkl` in a `model/` folder or add `tmdb_5000_movies.csv` to the project.')
        st.stop()

    # If similarity missing, create trivial identity similarity
    if similarity is None:
        try:
            n = len(movies)
            similarity = np.eye(n)
            st.info('Using trivial identity similarity matrix as fallback — recommendations will be basic.')
        except Exception as e:
            st.error(f'Could not create fallback similarity matrix: {e}')
            st.stop()

    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )

    if st.button('Show Recommendation'):
        recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
        with col2:
            st.text(recommended_movie_names[1])
            st.image(recommended_movie_posters[1])

        with col3:
            st.text(recommended_movie_names[2])
            st.image(recommended_movie_posters[2])
        with col4:
            st.text(recommended_movie_names[3])
            st.image(recommended_movie_posters[3])
        with col5:
            st.text(recommended_movie_names[4])
            st.image(recommended_movie_posters[4])
except Exception:
    tb = traceback.format_exc()
    # Print to server log
    print(tb)
    # Show the full traceback on the Streamlit page to help debugging on deploy
    st.error('Error running the app — full traceback shown below:')
    st.text(tb)


