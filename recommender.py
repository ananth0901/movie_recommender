import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, dataframe):
        self.movies = dataframe.copy()
        self._prepare_features()
        self._build_similarity_matrix()

    def _prepare_features(self):
        self.movies.fillna('', inplace=True)
        self.movies['features'] = (
            self.movies['Genre'] + ' ' +
            self.movies['Overview'] + ' ' +
            self.movies['Director'] + ' ' +
            self.movies['Star1'] + ' ' +
            self.movies['Star2'] + ' ' +
            self.movies['Star3'] + ' ' +
            self.movies['Star4']
        )

    def _build_similarity_matrix(self):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies['features'])
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        self.title_index = pd.Series(self.movies.index, index=self.movies['Series_Title']).drop_duplicates()

    def recommend(self, title, top_n=5):
        if title not in self.title_index:
            return []
        idx = self.title_index[title]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies['Series_Title'].iloc[movie_indices].tolist()
