# Movie Recommendation System

## Overview

This repository contains a Movie Recommendation System built using the TMDB dataset. It includes data files, a Jupyter notebook for development and analysis, and a Streamlit application for running the recommender.

## Key Files

-   `tmdb_5000_movies.csv`: Movie metadata.
-   `tmdb_5000_credits.csv`: Movie credits (cast, crew).
-   `movie_recommeder_system.ipynb`: Jupyter notebook for exploratory data analysis and model development.
-   `app.py`: The Streamlit application entry point.
-   `requirements.txt`: Python dependencies.
-   `.env`: Environment variables file (contains the TMDB API key).

## Step-by-step Workflow

### 1. Environment Setup

-   Create and activate a virtual environment:

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

-   Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### 2. Get an API Key

-   Create an account on [The Movie Database (TMDB)](https://www.themoviedb.org/).
-   Go to your account settings and generate an API key.
-   Create a file named `.env` in the root of the project and add your API key to it like this:

    ```
    TMDB_API_KEY=your_api_key
    ```

### 3. Generate the Model

-   Run the `movie_recommeder_system.ipynb` notebook to generate the `movies_dict.pkl` and `similarity.pkl` files. You can do this by running the following command:

    ```bash
    jupyter nbconvert --to notebook --execute movie_recommeder_system.ipynb
    ```

    Alternatively, you can run the notebook in a Jupyter environment.

### 4. Run the Application

-   Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

-   Open the local URL provided in your browser to use the movie recommender.

## Project Structure

```
.
├── .env
├── .gitignore
├── app.py
├── movie_recommeder_system.ipynb
├── Readme.md
├── requirements.txt
├── tmdb_5000_credits.csv
└── tmdb_5000_movies.csv
```

