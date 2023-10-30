from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from utils import evaluation
from torch import nn, optim
import pandas as pd
import numpy as np
import torch

class ALS(nn.Module):
    def __init__(self, usr_num, movie_num, dim):
        super().__init__()
        self.usr_emd = nn.Embedding(num_embeddings = usr_num, 
                                    embedding_dim = dim,
                                    max_norm = 1)
        
        self.movie_emd = nn.Embedding(num_embeddings = movie_num, 
                                      embedding_dim = dim,
                                      max_norm = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, usr, movie):
        usr =  self.usr_emd(usr)
        movie =  self.movie_emd(movie)
        mx = torch.sum(usr*movie, axis=1)
        return  self.sigmoid(mx)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):

        self.user = torch.tensor(df['new_user_id'].values)
        self.movie = torch.tensor(df['new_movie_id'].values)
        self.rating = torch.tensor(df['rating'].values, dtype=torch.float32)
 

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        return (self.user[idx], 
                self.movie[idx],
                self.rating[idx])
    

def training(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=10):
    
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        
        model.train()
        for _, (u, m, y) in enumerate(train_dataloader):

            y_pred = model(u, m)
            loss = loss_fn(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        for _, (u, m, y) in enumerate(test_dataloader):
            y_pred = model(u, m)
            loss = loss_fn(y_pred, y)
            test_loss.append(loss.item())

        print(f"Epoch:{epoch} | train_loss:{train_loss[-1]:.4f} | test_loss:{test_loss[-1]:.4f}")


    return model, train_loss, test_loss

if __name__ == '__main__':

    ## read data
    parser = ArgumentParser()
    parser.add_argument("-s", dest="sample", help="use smaller data set or not", type=bool, default=True)
    args = parser.parse_args()
    
    ## load data 
    if args.sample:
        data = pd.read_csv('data/small_rating.csv')
    else:
        data = pd.read_csv('data/procressed_rating.csv')
        
    print(f"sample:{args.sample}, data: {data.shape}")
    
    ## create like=1 (r=4~5)/ dislike=3 (r=1~3)
    data['class'] = data.rating.apply(lambda x: 1 if x >3 else 0)
    # print(data.head())

    ## train / test split
    print("=== train, test split")
    train, test= train_test_split(data, test_size=0.2, random_state=1126)
    print(f"train:{train.shape}, test:{test.shape}")
    # print(Dataset(train)[0])
    
    ## dataloader
    train_dataloader = DataLoader(Dataset(train), 
                                  batch_size=32, 
                                  shuffle=True)
    
    # raise ValueError()
    test_dataloader = DataLoader(Dataset(test), 
                                 batch_size=32)

    ## create als model
    model = ALS(usr_num=data.user_id.nunique(), 
                movie_num=data.movie_id.nunique(),
                dim = 16)
    print(model)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    ## model train & eval
    model, train_loss, test_loss = training(model, 
                                         train_dataloader, 
                                         test_dataloader, 
                                         loss_fn, 
                                         optimizer, 
                                         epochs=10)

    ## recommend?