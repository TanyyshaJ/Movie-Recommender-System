import streamlit as st
import pickle
import requests
import pandas as pd

api_key = 'YOUR_API_KEY_HERE'

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl','rb'))

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movies_posters



def fetch_poster(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/images?api_key={api_key}'

    # Make the GET request
    response = requests.get(url)

    if response.status_code == 200:
        images_data = response.json()

        # Extract poster path from the response
        posters = [image['file_path'] for image in images_data.get('posters', [])]

        # Construct full URLs for the posters
        base_image_url = 'https://image.tmdb.org/t/p/original/'
        poster_url = f"{base_image_url}{posters[0]}" 

        # Now poster_urls contains URLs of posters for the specified movie
        return poster_url


st.title('Movie Recommender System')
import streamlit as st

selected_movie_name = st.selectbox(
    'How would you like to be contacted?',
    movies['title'].values)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    columns = st.columns(5)

    for i in range(5):
        with columns[i]:
            if names and posters and i<len(names) and i<len(posters):
                st.text(names[i])
                st.image(posters[i]) 