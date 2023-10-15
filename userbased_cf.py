from tqdm import tqdm
from sortedcontainers import SortedList
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import collections

def dict_set(df, key, pair1, rate):
    result = defaultdict(dict)
    for _, row in df.iterrows():
        result[row[key]][row[pair1]] = float(row[rate])
    return result

def cal_similarity(r1, r2):
    common = set(r1.keys()) & set(r2.keys())
    m1, m2 = np.mean(list(r1.values())), np.mean(list(r2.values()))
    arr1 = [v-m1 for k, v in r1.items() if k in common]
    arr2 = [v-m2 for k, v in r2.items() if k in common]
    return (np.dot(arr1,arr2)) / (np.linalg.norm(arr1)*np.linalg.norm(arr2))


def training(train_data, k=25):
    neighbors, devs, avgs = defaultdict(list),  defaultdict(float),  defaultdict(float)
    for key, rate_results in tqdm(train_data.items()):
        
        # avg
        avg = np.mean(list(rate_results.values()))
        avgs[key]=avg

        # devs
        tmp_dev = {item: ratings-avg for item, ratings in rate_results.items()}
        devs[key] = tmp_dev

        # neighbors & weights
        tmp_neighgors = SortedList()
        for key1, rate_results1 in train_data.items():
            if key == key1: continue

            sim = cal_similarity(rate_results, rate_results1)
            tmp_neighgors.add((-sim, key1))
            if len(tmp_neighgors) > k: del tmp_neighgors[-1]

        neighbors[key].append(tmp_neighgors)

    return neighbors, devs, avgs

def predict_rate(data_dict, neighbors, devs, avgs, limit=5):
    results = []
    targets = []
    for key, value in data_dict.items():
        for item, rate in value.items():
            numerator = 0
            denominator = 0
            predict = avgs[key]
            if len(neighbors[key]) >= limit:
                for w, nei in neighbors[key].item():
                    w = -w
                    numerator += w * devs[key][nei]
                    denominator += abs(w)
                if denominator != 0: predict += numerator / denominator
            
            predict  = min(5, predict)
            predict  = max(0.5, predict) # min rating is 0.5
            targets.append(rate)
            results.append(predict)
    return results, targets

def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

if __name__ == '__main__':
    ## load data
    data = pd.read_csv('data/small_rating.csv')
    
    ## train / test split
    train, test= train_test_split(data, test_size=0.2, random_state=1126)
    print(f"train:{train.shape}, test:{test.shape}")

    ## get user-item-rate dataset
    train_dict = dict_set(train, key='userId', pair1='movieId', rate='rating')
    test_dict = dict_set(test, key='userId', pair1='movieId', rate='rating')

    # for key, value in train_dict.items():
    #     print(key, ":", value)
    #     break
    

    ## training: cal similarity(weight), avg, dev
    neighbors, devs, avgs = training(train_dict)
    print(f"neighbors: {len(neighbors)}, devs: {len(devs)}, avgs:{len(avgs)}")

    ## predict
    pre_train, tar_train = predict_rate(train_dict, neighbors, devs, avgs, limit=5)
    print('train mse:', mse(pre_train, tar_train))
    
    pre_test, tar_test = predict_rate(test_dict, neighbors, devs, avgs, limit=5)
    print('test mse:', mse(pre_test, tar_test))
