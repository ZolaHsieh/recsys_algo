from tqdm import tqdm
from argparse import ArgumentParser
from sortedcontainers import SortedList
from collections import defaultdict
from sklearn.model_selection import train_test_split
from utils import dict_set, cal_similarity, mse, evaluation
import pandas as pd
import numpy as np


def get_like_dislike(test_dict):
    like_list, dislike_list = defaultdict(set), defaultdict(set) 
    for item, ratings in tqdm(test_dict.items()):
        for usr, rate in ratings.items():
            if rate >=4: like_list[usr].add(item)
            else: dislike_list[usr].add(item)
    return like_list, dislike_list 


def training(train_data, k=10):
    neighbors, devs, avgs = defaultdict(list),  defaultdict(float),  defaultdict(float)
    for item, rate_results in tqdm(train_data.items()):
        
        # avg
        avg = np.mean(list(rate_results.values()))
        avgs[item]=avg

        # devs
        tmp_dev = {item: ratings - avg for item, ratings in rate_results.items()}
        devs[item] = tmp_dev

        # neighbors & weights
        tmp_neighgors = SortedList()
        for item1, rate_results1 in train_data.items():
            if item == item1: continue

            sim = cal_similarity(rate_results, rate_results1)
            tmp_neighgors.add((-sim, item1))
            if len(tmp_neighgors) > k: del tmp_neighgors[-1]

        neighbors[item] = tmp_neighgors
        
    return neighbors, devs, avgs


def predict_rate(data_dict, neighbors, devs, avgs, limit=5):
    results = []
    targets = []
    for item, ratings in data_dict.items():
        for _, rate in ratings.items():
            numerator = 0
            denominator = 0
            predict = avgs[item]
            if len(neighbors[item]) >= limit:
                for w, nei in neighbors[item]:
                    try: 
                        w = -w
                        numerator += w * devs[item][nei]
                        denominator += abs(w)
                    except KeyError: # user didn't watch movie that neighbot watch
                        pass
                if denominator != 0: 
                    predict += numerator / denominator
            
            predict  = min(5, predict)
            predict  = max(0.5, predict) # min rating is 0.5
            targets.append(rate)
            results.append(predict)
    return results, targets


def get_item_based_watched_list(data_set):
    watch_list = defaultdict(set)
    for item, ratings in data_set.items():
        for usr, _ in tqdm(ratings.items()):
            watch_list[usr].add(item)
        
    return watch_list


def gen_recommend(data_set, neighbors):
    recommend = defaultdict(set)
    watch_dict = get_item_based_watched_list(data_set)
    
    for usr, watched_list in tqdm(watch_dict.items()):
        neighbor_list = set()
        for item in watched_list:
            neighbor_list |= set(m for _, m in neighbors[item])
        recommend[usr] |= (neighbor_list - watched_list)

    return recommend


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-a", dest="all_data", help="use all data set or no", action="store_true") #default=False
    args = parser.parse_args()
    
    ## load data
    if not args.all_data:
        data = pd.read_csv('data/small_rating.csv')
    else: 
        data = pd.read_table('data/ratings.dat', 
                             sep='::',
                             header = None,
                             names=['user_id', 'movie_id', 'rating', 'timestamp'],
                             engine='python').drop('timestamp', axis =1)
        
    print(f"sample:{not args.all_data}, data: {data.shape}")
    
    ## train / test split
    print("=== train, test split")
    train, test= train_test_split(data, test_size=0.2, random_state=1126)
    print(f"train:{train.shape}, test:{test.shape}")

    ## get item-usr-rate dataset
    print("=== create item-usr-rate data set")
    train_dict = dict_set(train, key='movie_id', pair1='user_id', rate='rating')
    test_dict = dict_set(test, key='movie_id', pair1='user_id', rate='rating')

    ## create like, dislike list
    train_like, _ = get_like_dislike(train_dict)
    test_like, test_dislike =  get_like_dislike(test_dict)

    ## training: cal similarity(weight), avg, dev
    print("=== training")
    neighbors, devs, avgs = training(train_dict, k=20) # consdier 20 neighbors
    print(f"size: neighbors: {len(neighbors)}, devs: {len(devs)}, avgs:{len(avgs)}")
    
    ## predict
    print("=== predict rating")
    pre_train, tar_train = predict_rate(train_dict, neighbors, devs, avgs, limit=5)
    print('train mse:', mse(pre_train, tar_train))
    
    pre_test, tar_test = predict_rate(test_dict, neighbors, devs, avgs, limit=5)
    print('test mse:', mse(pre_test, tar_test))
    

    ## generate recommend movie list
    print("=== Generate recommend & cal precision, recall")
    recommend_list = gen_recommend(train_dict, neighbors)
    usr_list = set(test_dislike.keys() | test_like.keys())
    evaluation(usr_list, test_like, test_dislike, recommend_list)
