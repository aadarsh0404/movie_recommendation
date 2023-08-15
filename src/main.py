import pickle as pkl
import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup

# prepare data
movies = pd.read_csv('../dataset/movies.csv')
ratings = pd.read_csv('../dataset/ratings.csv')
data = movies.merge(ratings,on='movieId',how='left')
data.drop(['timestamp'],axis=1,inplace=True)
count = data.dropna(axis=0,subset=['title']).groupby('title')['rating'].count().reset_index().rename({'rating':'count'},axis=1)
data_combined = data.merge(count, on='title',how='left')
thresh = 50
popular_movie = data_combined.query("count>=@thresh" )
pivot_table = popular_movie.pivot_table(index='title',columns='userId',values='rating').fillna(0)

genres = []
for i in movies.genres:
    for j in i.split('|'):
        genres.append(j)
genres = list(set(genres))
pivot_table1 = pd.DataFrame(index=movies.title, columns=genres).fillna(0)

# Load Model
with open('../model/collaborative.pkl', 'rb') as f:
    model_collabrative = pkl.load(f)

with open('../model/content.pkl', 'rb') as f:
    model_content = pkl.load(f)

st.title('Movie Recommendation System')
option = st.selectbox(
    'How would you like to be contacted?',
    pivot_table1.index,
    placeholder="Type..."
)

suggest = st.button('Suggest Movies')
suggestions = []
if suggest:
    col1, col2 = st.columns(2)
    with col1:
        url = 'https://www.google.com/search?q={0}&tbm=isch'.format(option)
        content = requests.get(url).content
        soup = BeautifulSoup(content, 'lxml')
        images = soup.findAll('img')
        st.write('Recommendations for {0}:\n'.format(option))
    with col2:
        st.image(images[1].get('src'))

    query = list(pivot_table1.index).index(option)
    distances, indices = model_content.kneighbors(pivot_table1.iloc[query, :].values.reshape(1, -1), n_neighbors=6)
    for i in range(0, len(distances.flatten())):
        if i > 0:
            suggestions.append(pivot_table1.index[indices.flatten()[i]])

    try:
        query = list(pivot_table.index).index(option)
        distances, indices = model_collabrative.kneighbors(pivot_table.iloc[query, :].values.reshape(1, -1), n_neighbors=6)
        for i in range(0, len(distances.flatten())):
            if i > 0:
                suggestions.append(pivot_table.index[indices.flatten()[i]])
    except:
        pass

    for i in set(suggestions):
        col1, col2 = st.columns(2)
        with col1:
            url = 'https://www.google.com/search?q={0}&tbm=isch'.format(i)
            content = requests.get(url).content
            soup = BeautifulSoup(content, 'lxml')
            images = soup.findAll('img')
            st.write(i)
        with col2:
            st.image(images[1].get('src'))
