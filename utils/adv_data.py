import torch.utils.data as data

class AdvData(data.Dataset):
    def __init__(self, list_IDs, data, predicts):
        self.predicts = predicts
        self.list_IDs = list_IDs
        self.data = data
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self,index):
        ID = self.list_IDs[index]
                
        x = self.data[ID]
#        print(x.shape)
#        x = x.view(x.shape[1],x.shape[2],x.shape[3])
        y = self.predicts[ID]     
        return x, y

