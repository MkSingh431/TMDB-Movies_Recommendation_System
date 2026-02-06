import os
import pickle
import traceback

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")


def fetch_poster(movie_id):
    if not movie_id:
        return "https://via.placeholder.com/500x750?text=No+Image"
    url = (
        "https://api.themoviedb.org/3/movie/{}?api_key=0737e3e83144018bce73b00411bc39bd&language=en-US"
    ).format(movie_id)
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return "https://via.placeholder.com/500x750?text=No+Image"
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception:
        return "https://via.placeholder.com/500x750?text=No+Image"


@st.cache_data
def load_movies():
    # 1) model/movie_list.pkl
    movie_list_path = os.path.join(MODEL_DIR, "movie_list.pkl")
    if os.path.exists(movie_list_path):
        try:
            with open(movie_list_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    # 2) movies_dict.pkl (root)
    movies_dict_path = os.path.join(BASE_DIR, "movies_dict.pkl")
    if os.path.exists(movies_dict_path):
        with open(movies_dict_path, "rb") as f:
            data = pickle.load(f)
        return pd.DataFrame(data)

    # 3) CSV fallback
    csv_path = os.path.join(BASE_DIR, "tmdb_5000_movies.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "title" not in df.columns and "original_title" in df.columns:
            df = df.rename(columns={"original_title": "title"})
        return df

    return None


@st.cache_resource
def load_similarity(movies):
    # 1) model/similarity.pkl
    sim_path = os.path.join(MODEL_DIR, "similarity.pkl")
    if os.path.exists(sim_path):
        try:
            with open(sim_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    # 2) similarity.pkl (root)
    sim_path = os.path.join(BASE_DIR, "similarity.pkl")
    if os.path.exists(sim_path):
        try:
            with open(sim_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            # If this is a Git LFS pointer, ignore and fallback
            try:
                with open(sim_path, "rb") as f:
                    head = f.read(200)
                if b"git-lfs.github.com/spec" in head:
                    st.warning(
                        "Found a Git LFS pointer for `similarity.pkl` in deploy. "
                        "Either enable Git LFS in the repo or remove this file and "
                        "let the app build similarity from tags."
                    )
            except Exception:
                pass
            pass

    # 3) Build from tags if available
    if "tags" in movies.columns:
        cv = CountVectorizer(max_features=5000, stop_words="english")
        vectors = cv.fit_transform(movies["tags"]).toarray()
        return cosine_similarity(vectors)

    # 4) Fallback identity
    return np.eye(len(movies))


def recommend(movie, movies, similarity):
    if movies is None or similarity is None:
        return [], []

    idxs = movies[movies["title"] == movie].index
    if len(idxs) == 0:
        st.error(f'Movie "{movie}" not found in dataset.')
        return [], []

    index = idxs[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        row = movies.iloc[i[0]]
        if "movie_id" in movies.columns:
            movie_id = row.movie_id
        elif "id" in movies.columns:
            movie_id = row.id
        else:
            movie_id = None
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(row.title)

    return recommended_movie_names, recommended_movie_posters


try:
    st.header("Movie Recommender System")

    movies = load_movies()
    if movies is None:
        st.error(
            "Model files not found. Add `model/movie_list.pkl` or `movies_dict.pkl`, "
            "or add `tmdb_5000_movies.csv` to the project."
        )
        st.stop()

    similarity = load_similarity(movies)

    movie_list = movies["title"].values
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list,
    )

    if st.button("Show Recommendation"):
        recommended_movie_names, recommended_movie_posters = recommend(
            selected_movie, movies, similarity
        )
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
    print(tb)
    st.error("Error running the app — full traceback shown below:")
    st.text(tb)
