import numpy as np
import pandas as pd
import ast
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies=pd.read_csv("Database/movies.csv")
credit=pd.read_csv("Database/credits.csv")
# print( movies.head())
# print(credit.head())
# print(credit.head(1))

movies= movies.merge(credit,on="title")
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 

movies.dropna(inplace=True)
##print( movies)

movies['genres'] =   movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert3)
movies['cast'] =     movies['cast'].apply(convert)

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] =     movies['crew'].apply(fetch_director)



def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
##print(movies)

new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
new['tags'] = new['tags'].apply(lambda x: " ".join(x))

cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(new['tags']).toarray()

vector.shape

similarity = cosine_similarity(vector)

similarity
new[new['title'] == 'The Lego Movie'].index[0]
##print(similarity)

def recommend(movie):
    index = new[new['title'] == movie].index[0] 
##    print(index)
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    return distances



while True:
    entry = input("Enter Any Movie Name: ")
    if entry.lower() == "exit":
        break
    
    text=recommend(entry.title())
    for i in text[1:6]:

        ##it for movie name
        movie_title=new.iloc[i[0]].title
        print(movie_title)

