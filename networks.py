import torch
import torch.nn as nn
import torch.nn.functional as F

def set_model(model, device) :
    if model =='deep_narrow_resnet' :
        u_model = u_Net_deep_narrow_resnet().to(device)
    elif model == 'shallow_wide_resnet' :
        u_model = u_Net_shallow_wide_resnet().to(device)
    elif model == 'deep_narrow' :
        u_model = u_Net_deep_narrow().to(device)
    elif model == 'deep_narrow_1D' :
        u_model = u_Net_deep_narrow_1D().to(device)
    elif model == 'shallow_wide' :
        u_model = u_Net_shallow_wide().to(device)
    return u_model

class u_Net_shallow_wide(nn.Module):
    def __init__(self):
        super(u_Net_shallow_wide, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)    
#         self.fc3 = nn.Linear(256, 256)    
#         self.fc4 = nn.Linear(256, 256)    
        self.fc5 = nn.Linear(256, 1)
        self.act1 = nn.Tanh()
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act1(self.fc2(x))
#         x = self.act1(self.fc3(x))
#         x = self.act1(self.fc4(x))
        x = self.fc5(x)
        return x
    
    
class u_Net_shallow_wide_resnet(nn.Module):
    def __init__(self):
        super(u_Net_shallow_wide_resnet, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)    
#         self.fc3 = nn.Linear(256, 256)    
#         self.fc4 = nn.Linear(256, 256)    
        self.fc5 = nn.Linear(256, 1)
        self.act1 = nn.Tanh()
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act1(self.fc2(x))+x
#         x = self.act1(self.fc3(x))+x
#         x = self.act1(self.fc4(x))+x
        x = self.fc5(x)
        return x
    
class u_Net_deep_narrow(nn.Module):
    def __init__(self):
        super(u_Net_deep_narrow, self).__init__()
        self.fc1 = nn.Linear(2, 64)    #CHANGED TO SOLVE 1-D PROBLEMS
        #self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)    
        self.fc3 = nn.Linear(64, 64)    
        self.fc4 = nn.Linear(64, 64)  
        self.fc5 = nn.Linear(64, 64)  
        self.fc6 = nn.Linear(64, 64)  
        self.fc7 = nn.Linear(64, 64)  
        self.fc8 = nn.Linear(64, 64)  
        self.fc9 = nn.Linear(64, 1)
        self.act1 = nn.Tanh()
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act1(self.fc2(x))
        x = self.act1(self.fc3(x))
        x = self.act1(self.fc4(x))
        x = self.act1(self.fc5(x))
        x = self.act1(self.fc6(x))
        x = self.act1(self.fc7(x))
        x = self.act1(self.fc8(x))
        x = self.fc9(x)
        return x
    
class u_Net_deep_narrow_resnet(nn.Module):
    def __init__(self):
        super(u_Net_deep_narrow_resnet, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)    
        self.fc3 = nn.Linear(64, 64)    
        self.fc4 = nn.Linear(64, 64)  
        self.fc5 = nn.Linear(64, 64)  
        self.fc6 = nn.Linear(64, 64)  
        self.fc7 = nn.Linear(64, 64)  
        self.fc8 = nn.Linear(64, 64)  
        self.fc9 = nn.Linear(64, 1)
        self.act1 = nn.Tanh()
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act1(self.fc2(x))+x
        x = self.act1(self.fc3(x))+x
        x = self.act1(self.fc4(x))+x
        x = self.act1(self.fc5(x))+x
        x = self.act1(self.fc6(x))+x
        x = self.act1(self.fc7(x))+x
        x = self.act1(self.fc8(x))+x
        x = self.fc9(x)
        return x
    
class u_Net_deep_narrow_1D(nn.Module):
    """This is meant to solve 1-D problems"""
    def __init__(self):
        super(u_Net_deep_narrow_1D, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)
        self.act1 = nn.Tanh()
    def forward(self, x):
        y = self.act1(self.fc1(x))
        y = self.act1(self.fc2(y))
        y = self.fc3(y)
        print("x0:", x[0], "x1:", x[-1])
        #output = (1 - torch.exp(-(x - x[0])))*(1 - torch.exp(-(x[-1] - x))) * y 
        return y
    
