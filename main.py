from fastapi import FastAPI
import pickle
from recommender import MovieRecommender  # this line is 🔑

with open("movie_recommender.pkl", "rb") as f:
    recommender = pickle.load(f)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "🎬 Movie Recommender API is running!"}

@app.get("/recommend/{movie_title}")
def recommend_movies(movie_title: str):
    results = recommender.recommend(movie_title)
    if not results:
        return {"error": "Movie not found. Please check the title."}
    return {"recommendations": results}
