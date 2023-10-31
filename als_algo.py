from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from utils import eval
from torch import nn, optim
from tqdm import tqdm
import pandas as pd
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

    def forward(self, usr, movie):
        usr =  self.usr_emd(usr)
        movie =  self.movie_emd(movie)
        mx = torch.sum(usr*movie, axis=1)
        return  mx


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):

        self.user = torch.tensor(df['new_user_id'].values)
        self.movie = torch.tensor(df['new_movie_id'].values)
        self.rating = torch.tensor(df['class'].values, dtype=torch.float32)
 

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        return (self.user[idx], self.movie[idx], self.rating[idx])
    
# torch model training & eval
def training(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=10):
    
    train_losses = []
    test_losses = []
    for epoch in tqdm(range(epochs)):
        
        model.train()
        train_loss = 0
        train_p = train_r = train_acc = 0
        test_p = test_r = test_acc = 0
        for _, (u, m, y) in enumerate(train_dataloader):

            y_pred = model(u, m)
            loss = loss_fn(y_pred, y)
            train_loss +=loss.item()
            
            p, r, acc = eval(y, y_pred)
            train_p += p
            train_r += r
            train_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        test_loss = 0
        with torch.inference_mode():
            for _, (u, m, y) in enumerate(test_dataloader):
                y_pred = model(u, m)
                loss = loss_fn(y_pred, y)
                test_loss +=loss.item()

                p, r, acc = eval(y, y_pred)
                test_p += p
                test_r += r
                test_acc += acc

        train_losses.append(train_loss/len(train_dataloader))
        train_acc = train_acc/len(train_dataloader)
        train_p = train_p/len(train_dataloader)
        train_r = train_r/len(train_dataloader)

        test_losses.append(test_loss/len(test_dataloader))
        test_acc = test_acc/len(test_dataloader)
        test_p = test_p/len(test_dataloader)
        test_r = test_r/len(test_dataloader)

        print(f"Epoch:{epoch} | train - loss:{train_losses[-1]:.4f}, acc:{train_acc:.4f}, precision:{train_p:.4f}, recall:{train_r:.4f} "+
              f"| test_loss:{test_losses[-1]:.4f}, acc:{test_acc:.4f}, precision:{test_p:.4f}, recall:{test_r:.4f}")

    return model, train_loss, test_loss

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-a", dest="all_data", help="use all data set or no", action="store_true") #default=False
    args = parser.parse_args()
    
    ## load data
    if not args.all_data:
        data = pd.read_csv('data/small_rating.csv')
    else: 
        data = pd.read_csv('data/procressed_rating.csv')
    print(f"sample:{not args.all_data}, data: {data.shape}")
    
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
    test_dataloader = DataLoader(Dataset(test), batch_size=32)

    ## create als model
    model = ALS(usr_num=data.user_id.nunique(), 
                movie_num=data.movie_id.nunique(),
                dim = 16)
    print(model)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    ## model train & eval
    model, train_loss, test_loss = training(model, 
                                         train_dataloader, 
                                         test_dataloader, 
                                         loss_fn, 
                                         optimizer, 
                                         epochs=5)