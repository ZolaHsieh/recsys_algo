from argparse import ArgumentParser
from collections import Counter
import pandas as pd

def reindex_id(df, col: str):
    id_list = list(df[col].unique())
    cnt = [i for i in range(len(id_list))]
    mapp = dict(zip(id_list, cnt))
    print(f'reindex: {len(cnt)}')
    return df[col].map(mapp)

def create_small_ratings(usr_num, movie_num, rating_data):

    print(f"create smaller dataset: usr_num={usr_num}, moive_num={movie_num}")

    user_cntr = Counter(rating_data.user_id)
    most_often_user = [u for u, _ in user_cntr.most_common(usr_num)]

    movie_cntr = Counter(rating_data.movie_id)
    most_often_movie = [m for m, _ in movie_cntr.most_common(movie_num)]

    smaller_rating = rating_data[rating_data.user_id.isin(most_often_user) & \
                               rating_data.movie_id.isin(most_often_movie)].reset_index(drop=True)

    print('rating dataset: ',rating_data.shape,' -> ' , smaller_rating.shape)

    ## to _csv
    smaller_rating.to_csv('data/small_rating.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-n", dest="usr_num", help="number of users for sampling", type=int, default=1000)
    parser.add_argument("-m", dest="moive_num", help="number of moive for sampling", type=int, default=200)
    args = parser.parse_args()

    rating_data = pd.read_table('data/ratings.dat', 
                         sep='::',
                         header = None, 
                         names=['user_id', 'movie_id', 'rating', 'timestamp'],
                         engine='python').drop('timestamp', axis =1)
    
    print(f"rating data: {rating_data.shape}")
    print(f'user_id min:{rating_data.user_id.min()}, max:{rating_data.user_id.max()}, cnt:{rating_data.user_id.nunique()}')
    print(f'movie_id min:{rating_data.movie_id.min()}, max:{rating_data.movie_id.max()}, cnt:{rating_data.movie_id.nunique()}')
    # print(rating_data.head())


    movie_data = pd.read_table('data/movies.dat', 
                            sep='::',
                            header = None, 
                            names=['movie_id', 'title', 'genres'],
                            engine='python',
                            encoding='ISO-8859-1')
    print(f"movie data: {movie_data.shape}")
    print(f'movie_id min:{movie_data.movie_id.min()}, max:{movie_data.movie_id.max()}, cnt:{movie_data.movie_id.nunique()}')
    # print(movie_data.head())


    usr_data = pd.read_table('data/users.dat', 
                            sep='::',
                            header = None, 
                            names=['user_id', 'gender', 'age', 'occupation', 'zip'],
                            engine='python')
    print(f"usr data: {usr_data.shape}")
    print(f'user_id min:{usr_data.user_id.min()}, max:{usr_data.user_id.max()}, cnt:{usr_data.user_id.nunique()}')
    # print(usr_data.head())
    

    ## movie data preprocess
    one_hot_genres = movie_data.genres.str.strip().str.get_dummies(sep = '|')
    movie_data = pd.concat([movie_data, one_hot_genres], axis=1)
    print(f'procressed movie:{movie_data.shape}')
    movie_data.to_csv('data/procressed_movie.csv', index=False)
    
    ## usr data preprocess
    usr_data.gender = usr_data.gender.map({'F':0, 'M':1})
    usr_data.age = usr_data.age // 10
    usr_data = pd.get_dummies(usr_data,prefix=['occupation'], 
                              columns = ['occupation'], 
                              drop_first=True, 
                              dtype=int)
    print(f'procressed usr:{usr_data.shape}')
    usr_data.to_csv('data/procressed_usr.csv', index=False)

    ## create smaller dataset
    create_small_ratings(args.usr_num, args.moive_num, rating_data)