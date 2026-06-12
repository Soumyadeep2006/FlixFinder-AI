import streamlit as st
import pickle
import requests

API_KEY = "4841f5e9"

def fetch_poster(movie_name):
    try:
        url = f"https://www.omdbapi.com/?t={movie_name}&apikey={API_KEY}"
        data = requests.get(url).json()

        if data.get("Poster") and data["Poster"] != "N/A":
            return data["Poster"]

    except:
        pass

    return None

# Load Data
movies = pickle.load(
    open('models/movies.pkl', 'rb')
)

similarity = pickle.load(
    open('models/similarity.pkl', 'rb')
)

# Recommendation Function
def recommend(movie):

    movie_index = movies[
        movies['title'] == movie
    ].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i in movie_list:

        movie_data = movies.iloc[i[0]]

        recommendations.append(
            (
                movie_data.title,
                round(movie_data.vote_average, 1),
                round(i[1] * 100, 2),
                movie_data.overview_text,
                ", ".join(movie_data.genres)
            )
        )

    return recommendations


# Page Configuration
st.set_page_config(
    page_title="FlixFinder-AI",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
with st.sidebar:

    st.header("📊 Project Information")

    st.write(
        "Movie Recommendation System using TF-IDF and Cosine Similarity."
    )

    st.metric(
        "Movies Available",
        len(movies)
    )

    st.metric(
        "Algorithm",
        "TF-IDF + N-Grams"
    )

    st.metric(
        "Similarity",
        "Cosine Similarity"
    )

st.divider()

st.subheader("🏆 Top Rated Movies")

top_movies = movies[
    (movies["vote_average"] >= 7.0) &
    (movies["vote_count"] >= 1000)
].sort_values(
    by="vote_average",
    ascending=False
).head(5)

cols = st.columns(5)

for idx, (_, row) in enumerate(top_movies.iterrows()):

    with cols[idx]:

        poster = fetch_poster(row["title"])

        if poster:
            st.image(poster, use_container_width=True)

        st.markdown(
            f"**{row['title']}**"
        )

        st.caption(
            f"⭐ Rating: {round(row['vote_average'],1)}"
        )

# Main Title
st.markdown("# 🤖 FlixFinder-AI")

st.markdown(
    "### Your next favorite movie is just one click away."
)

st.divider()

selected_movie = st.selectbox(
    "🔍 Search or Select Movie",
    movies['title'].values
)

if st.button("🍿 Find My Next Movie"):

    recommendations = recommend(
        selected_movie
    )

    st.subheader(
        "🎥 Recommended Movies"
    )

    cols = st.columns(5)

    for idx, (
    movie,
    rating,
    match_score,
    overview,
    genres
) in enumerate(recommendations):
    

        with cols[idx]:

            poster = fetch_poster(movie)

            if poster:
                st.image(poster, use_container_width=True)

            st.markdown(
            f"### 🎬 {movie}"
        )

            st.write(
            f"⭐ Rating: {rating}"
        )

            if match_score >= 40:

                st.success(
                "🏆 A+ Recommendation"
            )

            elif match_score >= 35:

                st.success(
                "⭐ A Recommendation"
            )

            elif match_score >= 30:

                st.info(
                "✨ B+ Recommendation"
            )

            elif match_score >= 20:

                st.info(
                "👍 B Recommendation"
            )

            else:

                st.warning(
                "🎬 Worth Watching"
            )

        with st.expander(
            "📖 Read Overview"
        ):
            st.write(
                overview
            )

        st.write(
            f"🎭 Genres: {genres}"
        )

        st.divider()


st.divider()

st.caption(
    "Built with Streamlit • TF-IDF • Cosine Similarity • TMDB 5000 Dataset"
)
