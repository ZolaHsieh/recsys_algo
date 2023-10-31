from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from utils import eval
from torch import nn, optim
from tqdm import tqdm
import pandas as pd
import torch

class LR(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.linear_layer = nn.Linear(in_features=dim, out_features=1)
        nn.init.xavier_uniform_(self.linear_layer.weight)  # 使用Xavier初始化

    def forward(self, x):
        return self.linear_layer(x)


class Poly2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.w0 = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(1, 1)))  
        self.w1 = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(dim, 1)))  
        self.w2 = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(dim, dim)))  

    def __cross(self, x):
        x_left = torch.unsqueeze(x, 2) #[ batch_size, n_feats, 1 ]
        x_right = torch.unsqueeze(x, 1) # [ batch_size, 1, n_feats ]
        x_cross = torch.matmul(x_left, x_right) # [ batch_size, n_feats, n_feats ]

        cross_out = torch.sum(torch.sum(x_cross * self.w2, dim=2), dim=1, keepdim=True)  # [ batch_size, 1 ]
        return cross_out

    def forward(self, x):
        a = self.w0 + torch.matmul(x, self.w1)
        b = self.__cross(x)
        return a + b


class FM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.w0 = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(1, 1)))  
        self.w1 = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(dim, 1)))  
        self.w2 = nn.init.xavier_uniform_(nn.Parameter(torch.Tensor(dim, dim)))  #v

    def __FMcross(self, x):
        
        a = torch.torch.matmul(x, self.w2)**2 #[batch, dim]
        b = torch.torch.matmul(x**2, self.w2**2) #[batch, dim]
        out = 0.5 * torch.sum(a-b, dim=1, keepdim=True) #[batch, 1]
        return out

    def forward(self, x):
        a = self.w0 + torch.matmul(x, self.w1)
        b = self.__FMcross(x)
        return a + b


class FM_EMB(nn.Module):
    def __init__(self, num_feat, emb_dim):
        super().__init__()
        
        self.xv = nn.Embedding(num_embeddings = num_feat, 
                               embedding_dim = emb_dim,
                               max_norm = 1)  # serve xv as an emb

    def __FMcross(self, emb):
        
        a = torch.torch.sum(emb, dim=1)**2 # square_of_sum [batch, num_feat, emb_dim] -> [batch, emb_dim]
        b = torch.torch.sum(emb**2, dim=1) #sum of square [batch, num_feat, emb_dim] -> [batch, emb_dim]
        out = 0.5 * torch.sum(a-b, dim=1, keepdim=True) #[batch, 1]
        return out

    def forward(self, x):
        emb = self.xv(x)
        return self.__FMcross(emb)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, fmb=False):

        if not fmb:
            self.x = torch.tensor(df.drop('class', axis=1).values, dtype=torch.float)
        else: 
            self.x = torch.tensor(df.drop('class', axis=1).values) # if use embedding, type should be LongTensor
        self.y = torch.tensor(df['class'].values, dtype=torch.float)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
    

# torch model training & eval
def training(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=10):
    
    train_losses = []
    test_losses = []
    for epoch in tqdm(range(epochs)):
        
        model.train()
        train_loss = 0
        train_p = train_r = train_acc = 0
        test_p = test_r = test_acc = 0
        for _, (X, y) in enumerate(train_dataloader):

            y_pred = model(X).squeeze(dim=1)
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
            for _, (X, y) in enumerate(test_dataloader):
                y_pred = model(X).squeeze(dim=1)
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
    parser.add_argument("-e", dest="epochs", type=int, default=5)
    parser.add_argument("-m", dest="mdl_type", type=str, default="lr")
    args = parser.parse_args()
    
    ## load rating datadata
    if not args.all_data:
        data = pd.read_csv('data/small_rating.csv')
    else: 
        data = pd.read_csv('data/procressed_rating.csv')
    print(f"sample:{not args.all_data}, data: {data.shape}")
    data['class'] = data.rating.apply(lambda x: 1 if x >3 else 0) ## create like=1 (r=4~5)/ dislike=3 (r=1~3)

    ## user features & item features load and merge
    usr_data = pd.read_csv('data/procressed_usr.csv')
    movie_data = pd.read_csv('data/procressed_movie.csv')

    data = data.merge(usr_data, on =['user_id'], how='left')\
                .merge(movie_data, on =['movie_id'], how='left')\
                .drop(['user_id', 'movie_id', 'new_user_id', 'new_movie_id', 'title', 'genres','zip', 'rating'],axis=1)
    # print(data.columns, data.head())
    
    ## train / test split
    print("=== train, test split")
    train, test= train_test_split(data, test_size=0.2, random_state=1126)
    print(f"train:{train.shape}, test:{test.shape}")
    # print(Dataset(train)[0])
    
    ## dataloader
    train_dataloader = DataLoader(Dataset(train, fmb=args.mdl_type=='fm2'), 
                                  batch_size=32, 
                                  shuffle=True)
    
    test_dataloader = DataLoader(Dataset(test, fmb=args.mdl_type=='fm2'), 
                                 batch_size=32)
    ##### create LR model #####
    if args.mdl_type=='lr':
        model = LR(dim = train.shape[1]-1)
    if args.mdl_type=='poly':
        model = Poly2(dim = train.shape[1]-1)
    if args.mdl_type=='fm1':
        model = FM(dim = train.shape[1]-1)
    if args.mdl_type=='fm2':
        model = FM_EMB(num_feat = train.shape[1]-1,
                       emb_dim=128)
    
    print(model)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    ## model train & eval
    model, train_loss, test_loss = training(model, 
                                         train_dataloader, 
                                         test_dataloader, 
                                         loss_fn, 
                                         optimizer, 
                                         epochs=args.epochs)