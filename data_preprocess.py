from collections import Counter
import pandas as pd

def reindex_id(df, col: str):
    id_list = list(df[col].unique())
    cnt = [i for i in range(len(id_list))]
    mapp = dict(zip(id_list, cnt))
    print(f'reindex: {len(cnt)}')
    return df[col].map(mapp)


data = pd.read_csv('data/rating.csv').drop('timestamp', axis =1)
print(f'userID min:{data.userId.min()}, max:{data.userId.max()}, cnt:{data.userId.nunique()}')
print(f'movieId min:{data.movieId.min()}, max:{data.movieId.max()}, cnt:{data.movieId.nunique()}')
# print(data.head())

## user id, movie id transform
data.userId = data.userId - 1
data.movieId = reindex_id(data, 'movieId')
# print(data.head())

## to _csv
data.to_csv('data/new_rating.csv', index=False)


## create smaller dataset
# number of users and movies we would like to keep
n = 10000
m = 2000

user_cntr = Counter(data.userId)
most_often_user = [u for u, _ in user_cntr.most_common(n)]

movie_cntr = Counter(data.movieId)
most_often_movie = [m for m, _ in movie_cntr.most_common(m)]

smaller_data = data[data.userId.isin(most_often_user)&data.movieId.isin(most_often_movie)].reset_index(drop=True)
print('create smaller dataset: ',data.shape,' -> ' , smaller_data.shape)
# print(smaller_data.head())

## reindex user id and movie id
smaller_data.userId = reindex_id(smaller_data, 'userId')
smaller_data.movieId = reindex_id(smaller_data, 'movieId')
# print(smaller_data.head())

## to _csv
smaller_data.to_csv('data/small_rating.csv', index=False)