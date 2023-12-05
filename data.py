#Importing Libraries
import pandas as pd
import numpy as np
import os

movies = pd.read_csv("tmdb_5000_movies.csv")
credit = pd.read_csv("tmdb_5000_credits.csv")

#Merging 2 datasets
movies = movies.merge(credit, on="title")
#print(movies.info()) - checking info to remove
#Preprocessing phase
#Removing unwanted columns
#Columns to be kept: genres, id, keywords, title, overview, cast, crew

movies = movies[['genres', 'id', 'keywords','title', 'overview', 'cast', 'crew']]

#print(movies.info()) - checking final info
#print(movies.isnull().sum()): gives the number of null elements(if any) for a particular column

movies.dropna(inplace=True) #drops null objects
# print(movies.duplicated().sum()): to check if there are any duplicate rows

#genres column is a string of list of dictionaries, we only wan the genre names in a list format

#ast is a module in python whose literal_eval function converts this striing into list
import ast
#this function only returns a list of genres of the movie
def convert(obj):
    lst = []
    for i in ast.literal_eval(obj):
        lst.append(i['name'])
    return lst

#applying the function on genres column of movies
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

#Gets the name of top 3 cast actors only
def convert3(obj):
    lst = []
    counter = 0
    for i in ast.literal_eval(obj):
        if(counter!=3):
            lst.append(i['name'])
            counter+=1
        else:
            break
    return lst

movies['cast'] = movies['cast'].apply(convert3)

#function that extracts the name of the director
def getDirector(obj):
    lst = []
    for i in ast.literal_eval(obj):
        if(i['job'] == 'Director'):
            lst.append(i['name'])
            
    return lst

movies['crew'] = movies['crew'].apply(getDirector)

#Converting movies['overview'] col which is a string into a list -> makes it easy to concatenate
movies['overview'] = movies['overview'].apply(lambda x: x.split())

#It is crucial to remove whitespaces between the name and surname of cast and crew members, so that when we create tags, they are of the full name 'Varun Dhawan' would create 2 tags: 'Varun' and 'Dhawan' and to avoid confusion with any other actor of the same name, we do this conversion: 'VarunDhawan', creates a single tag!

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])

#Concatenating columns: overview, genres, keywords, cast, crew into a new column named "tags"

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

#Removing duplicates from tags and making them unique. Making a new dataframe, not changing it in the main movies
new_df = movies[['id', 'title', 'tags']]

#The contents of tags column is in a list format, we now convert it back into string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
#Converting everything to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


#problem: there some words repeated but in different forms: activity and activites, they take up 2 spaces! Thus, we apply stemming, we get 1 root word for all words. For this, use nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

#Now, to find/recommend similar movies to the user, we convert the tags(text) to vector and then return the closest 5 vectors. Do not consider 'stop words'-> only contribute in sentence formation, do not add meaning
#We use: Bag of words
#1.Concatenate all tags    2.Extract top 5000 common words    3.Count the frequency of these words in tags of all movies

#using scikit-learn library
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vector_tuple = cv.fit_transform(new_df['tags'])
vectors = vector_tuple.toarray()     #movie tags converted to vectors
#to know which the top 5000 words are: cv.get_feature_name()
#print(cv.get_feature_names_out())

#Now that we have vectors, calculate distance of each vector with every other vector! Greater distance = Less similarity Calculating: cosine distance (smaller angle: less distance = more similar)
#High-dim data: we use cosine distance, Euclidean distance fails!

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

#define a function, when i input a movie name, it returns 5 similar movies
#1. Find id of the inputted movie name  2. In the similarity array, sort the distances, and find top 5 least distances, fetch these movies and return them

def recommend(movie):
    #Fetching title
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]

    #problem: when we sort, the indices of the movies are lost. To hold indices of movies, we call enumerate function and convert it into list
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

import pickle
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity,open('similarity.pkl', 'wb'))