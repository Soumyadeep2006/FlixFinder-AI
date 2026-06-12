import pandas as pd
import pickle
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Required columns
movies = movies[
    [
        'movie_id',
        'title',
        'overview',
        'genres',
        'keywords',
        'cast',
        'crew',
        'vote_average',
        'vote_count'
    ]
]

movies.dropna(inplace=True)


# Convert JSON columns
def convert(text):
    result = []

    for item in ast.literal_eval(text):
        result.append(item['name'])

    return result


def convert_cast(text):
    result = []

    for item in ast.literal_eval(text)[:5]:
        result.append(item['name'])

    return result


def fetch_director(text):
    result = []

    for item in ast.literal_eval(text):
        if item['job'] == 'Director':
            result.append(item['name'])

    return result


# Apply transformations
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(
    lambda x: x.split()
)

# Save readable overview
movies['overview_text'] = movies['overview'].apply(
    lambda x: " ".join(x)
)

# Remove spaces except genres
movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

movies['crew'] = movies['crew'].apply(
    lambda x: [i.replace(" ", "") for i in x]
)

# Weighted tags
movies['tags'] = (
    movies['overview']
    + movies['genres'] * 3
    + movies['keywords'] * 2
    + movies['cast'] * 2
    + movies['crew'] * 3
)

movies['tags'] = movies['tags'].apply(
    lambda x: " ".join(x).lower()
)

# Final dataframe
new_df = movies[
    [
        'movie_id',
        'title',
        'overview_text',
        'genres',
        'vote_average',
        'vote_count',
        'tags'
    ]
]

# Improved TF-IDF
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2)
)

vectors = tfidf.fit_transform(
    new_df['tags']
)

similarity = cosine_similarity(
    vectors
)

# Save models
pickle.dump(
    new_df,
    open('models/movies.pkl', 'wb')
)

pickle.dump(
    similarity,
    open('models/similarity.pkl', 'wb')
)

pickle.dump(
    tfidf,
    open('models/tfidf.pkl', 'wb')
)

print("Model saved successfully")