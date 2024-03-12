from tqdm import tqdm
import pickle as pkl
import numpy as np
import copy
import argparse
import sys
sys.path.append("..")

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from networks import set_model, u_Net_shallow_wide, u_Net_shallow_wide_resnet, u_Net_deep_narrow, u_Net_deep_narrow_resnet

import matplotlib.pyplot as plt
def q(data):
    return (np.pi**2) * torch.sin(np.pi * data)

def analytic(data):
    return torch.sin(np.pi * data)


def calculate_derivative(y, x) :
    return torch.autograd.grad(y, x, create_graph=True,\
                        grad_outputs=torch.ones(y.size()).to(device))[0]


def calculate_all_partial(u, x) :
    del_u = calculate_derivative(u, x)
    u_x= del_u[:,0]
    u_xx = calculate_derivative(u_x, x)[:,0]
    return u_xx.view(-1,1) #, u_yy.view(-1,1)



def train(u_model, beta, lbd, trainloader, bdry_data, val_test, optimizer, loss_f) :
    loss_list, loss_list1, loss_list2, val_list, test_list = [], [], [], [], []
    X_bdry, u_bdry = bdry_data
    X_val, y_val, X_test, y_test = val_test

    for i, (data,) in enumerate(trainloader) :
        u_model.train()
        optimizer.zero_grad()
        X_v = Variable(data, requires_grad=True).to(device)
        output = u_model(X_v)  
        output_bdry = u_model(X_bdry)
        
        u_xx = calculate_all_partial(output, X_v)
        loss1 = loss_f(u_xx + q(X_v), torch.zeros_like(output))
        loss2 = loss_f(output_bdry, torch.zeros_like(output_bdry))
        
        loss = loss1 + beta*loss2 + (lbd*output_bdry.view(-1)).mean()
        loss.backward()
        lbd.grad *= -1
        
        optimizer.step()
        
        u_model.eval()
        val_err = torch.linalg.norm((u_model(X_val) - y_val),2).item() / torch.linalg.norm(y_val,2).item()
        test_err = torch.linalg.norm((u_model(X_test) - y_test),2).item() / torch.linalg.norm(y_test,2).item()

        loss_list.append((loss1+loss2).item())
        loss_list1.append(loss1.item())
        loss_list2.append(loss2.item())
        val_list.append(val_err)
        test_list.append(test_err)
        
    return np.mean(loss_list), np.mean(loss_list1), np.mean(loss_list2), np.mean(val_list), np.mean(test_list)


def main_function(model_name, beta, lr, lbd_lr, EPOCH, device) :
    
    # Dataset Creation
    xmin, xmax = 0,1
    ymin, ymax = -1,1
    Nx = 51
    #X_train = torch.FloatTensor(np.mgrid[xmin:xmax:51j, ymin:ymax:51j].reshape(2, -1).T).to(device)
    X_train = torch.FloatTensor(np.linspace(xmin, xmax, Nx)).view(-1, 1).to(device)

    # Boundary Conditions
    #X_bdry = X_train[(X_train[:,0]==xmin) + (X_train[:,0]==xmax) + (X_train[:,1]==ymin) + (X_train[:,1]==ymax)]
    #u_bdry = torch.zeros_like(X_bdry[:,0]).to(device).view(-1,1)
        
    X_bdry = torch.FloatTensor([0, 1]).view(-1, 1).to(device)
    u_bdry = torch.zeros_like(X_bdry)

    X_test, y_test, X_val, y_val= torch.load('Poisson_test', map_location=device)
    
    X_test, X_val = X_test[:,0].reshape(-1, 1), X_val[:, 0].reshape(-1, 1)


    print("x_test shape: ", X_test.shape, " y_test shape: ", y_test.shape)


    # Make dataloader
    data_train = TensorDataset(X_train)
    train_loader = DataLoader(data_train, batch_size=10000, shuffle=False)
    
    # train
    total_loss, test_errs, val_errs = [], [], []
    u_model = set_model(model_name, device)
    lbd = Variable(torch.FloatTensor([0]*X_bdry.size()[0]).to(device), requires_grad=True)
    
    optimizer=torch.optim.Adam([{'params': u_model.parameters()}, {'params': lbd, 'lr':lbd_lr}], lr=lr)
    best_model = copy.deepcopy(u_model)

    for t in tqdm(range(0, EPOCH)) :

        loss, loss1, loss2, val_err, test_err = train(u_model, beta, lbd, trainloader=train_loader,\
                                                      bdry_data=[X_bdry, u_bdry],\
                                                      val_test = [X_val, y_val, X_test, y_test],\
                                                      optimizer=optimizer, loss_f=nn.MSELoss())
        
        val_errs.append(val_err)
        test_errs.append(test_err)  
        total_loss.append(loss)
        
#         # Print Log
#         if t%100 == 0 :
#             print("%s/%s | loss: %06.6f | loss_f: %06.6f | loss_u: %06.6f | val error : %06.6f | test error : %06.6f " % \
#                   (t, EPOCH, loss, loss1, loss2, val_err, test_err))

        if np.argmin(val_errs) == t :
            best_model = copy.deepcopy(u_model)

    return best_model, total_loss, val_errs, test_errs
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deep_narrow_1D', help='Specify the model. Choose one of [deep_narrow, shallow_wide, deep_narrow_resent, shallow_wide_resnet].')
    parser.add_argument('--beta', default=5000, type=float, help='Penalty parameter beta')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--lbd_lr', default=1, type=float, help='Learning rate for lambda')
    parser.add_argument('--EPOCH', default=5000, type=int, help='Number of training EPOCH')
    parser.add_argument('--ordinal', default=0, type=int, help='Specify the cuda device ordinal.')
    args = parser.parse_args()
    
    device = torch.device("cuda:{}".format(args.ordinal) if torch.cuda.is_available() else "cpu")
    
    best_model, total_loss, val_errs, test_errs = main_function(args.model, args.beta, args.lr, args.lbd_lr, args.EPOCH, device)
    
    
        # Assuming 'x' is your domain tensor and 'model' is your trained model
    x = torch.linspace(0, 1, steps=1000).view(-1, 1).to(device)  # Ensure 'x' is on the correct device

    # Ensure the model is in evaluation mode
    best_model.eval()

    # Use the model to predict the solution
    with torch.no_grad():  # No need to track gradients here
        predicted_solution = best_model(x)

    # Convert predictions and x to CPU for plotting
    predicted_solution = predicted_solution.cpu().numpy()
    x = x.cpu().numpy()

    # Exact solution for comparison
    exact_solution = np.sin(np.pi * x)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, predicted_solution, label='Predicted Solution', linestyle='--', color='blue')
    plt.plot(x, exact_solution, label='Exact Solution', linestyle='-', color='red')
    plt.xlabel('Domain (x)')
    plt.ylabel('Solution (u)')
    plt.title('Comparison of Predicted and Exact Solutions for the Poisson Equation')
    plt.legend()
    plt.show()
    
    
    print('Best Test Error : ', test_errs[np.argmin(val_errs)])
    