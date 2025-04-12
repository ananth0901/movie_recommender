import pandas as pd
import pickle
from recommender import MovieRecommender

df = pd.read_csv("data/imdb.csv")
recommender = MovieRecommender(df)

with open("movie_recommender.pkl", "wb") as f:
    pickle.dump(recommender, f)

print("✅ Model trained and saved to movie_recommender.pkl")
