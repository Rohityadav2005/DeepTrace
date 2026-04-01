import torch.nn as nn 

class ResNetLSTM(nn.Module):
    def __init__(self,resnet_base,hidden_dim = 512,num_layers = 2):
        super(ResNetLSTM,self).__init__()
        self.resnet = resnet_base

        self.lstm = nn.LSTM(input_size=2048,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_dim,2)
                              
    def forward(self,x):
        batch_size,seq_len,c,h,w = x.shape

        x = x.view(batch_size*seq_len,c,h,w)
        features = self.resnet(x)
        features = features.view(batch_size,seq_len,-1)
        lstm_out,(hidden,cell) = self.lstm(features)
        last_frame_feat = lstm_out[:,-1,:]
        return self.fc(last_frame_feat)
    
    