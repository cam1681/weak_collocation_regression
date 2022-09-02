"""
Function: data-driven reveal SDE by the weak form of FKE: approximation of coefficients by neural networks
@author pi square 
@email: hpp1681@gmail.com
created in Oct 28, 2021
update log:
    0. 10/28/2021: created 
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import time
import utils
import scipy.io

device = torch.device("cpu")
torch.manual_seed(1234)

class DNN(torch.nn.Module):
    def __init__(self, layers = [3]+[10]*3+[1]):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.tanh = torch.nn.Tanh
        self.relu = torch.nn.ReLU
        self.softplus = torch.nn.Softplus
        self.sigmoid = torch.nn.Sigmoid

        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.sigmoid()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out




class Model(object):
    """A ``Model`` solve the true coefficients of the basis on the data by the outloop for linear regression and 
    and the inner loop of increasing the parameters in the test function TestNet.
    Args:
        t : `` t'' vector read from the file
        data: ``data`` matrix read from the file.
        testFunc: ``DNN`` instance.
    """
    def __init__(self, t, data, psi, phi1, phi2):
        self.t = t
        self.itmax = len(t)
        self.data = data
        self.net_psi = psi
        self.net_phi1 = phi1
        self.net_phi2 = phi2
        
        self.t_number = data.shape[0]
        self.bash_size = data.shape[1]
        self.dimension = data.shape[2]


        self.error_tolerance = None
        self.max_iter = None
        self.loss = None

        
        # self.batch_size = None # tbd
        # self.train_state = TrainState() # tbd
        # self.losshistory = LossHistory() # tbd


    def _get_data_t(self, it):
        X = self.data[it,:,:]
        return X
                
    @utils.timing
    def compile(self, step_test=10, step_neural=10, error_tolerance=1e-3, max_iter=1e5, optimizer="SGD"):  
        params = list(self.net_psi.parameters()) + list(self.net_phi1.parameters()) + list(self.net_phi2.parameters())
        self.optimizer = utils.optimizer_get(params, optimizer, learning_rate=-1e-3)
        self.error_tolerance = error_tolerance
        self.max_iter = max_iter
        self.step_test = step_test
        self.step_neural = step_neural


    @torch.enable_grad()
    def compute_loss(self):
        rb = torch.zeros(self.t_number)
        self.b = torch.zeros(self.t_number)


        # ##########################################################
        #  Tensor form of computing A and b for parallel computing
        # ##########################################################

        TX = self.data
        TX.requires_grad = True
        out_psi = self.net_psi(TX)
        out_phi1 = self.net_phi1(TX)
        out_phi2 = self.net_phi2(TX)
        
        F1 = 1/self.bash_size * torch.sum(
            torch.autograd.grad(
                outputs=out_psi, inputs=TX, 
                grad_outputs=torch.ones_like(out_psi),
                create_graph=True,
                retain_graph=True)[0][:,:,:] * 
            torch.autograd.grad(
                outputs=out_phi1, inputs=TX, 
                grad_outputs=torch.ones_like(out_phi1),
                create_graph=True,
                retain_graph=True)[0][:,:,:]
            ,
            dim = 2
            )
        
        F2 = torch.zeros(self.t_number)
        for ld in range(self.dimension):
            for kd in range(self.dimension):
                output=torch.autograd.grad(
                            outputs=out_psi, inputs=TX, 
                            grad_outputs=torch.ones_like(out_psi),
                            create_graph=True,
                            retain_graph=True
                            )[0][:,:,ld]
                F2 = F2 + 1/self.bash_size * torch.sum(
                    torch.autograd.grad(
                        outputs=output,
                        inputs=TX,
                        grad_outputs=torch.ones_like(output),
                        create_graph=True,
                        retain_graph=True
                        )[0][:,:,kd] * 
                    out_phi2[:,:,ld] *
                    out_phi2[:,:,kd]
                    ,
                    1
                    )
        self.F = F1 + F2
                
        rb = 1/self.t_number * torch.sum(out_psi, dim=1).squeeze()
        dt = (torch.max(self.t)-torch.min(self.t)) / self.t_number
        def compute_ut(u, n, dt):   
            ut = torch.zeros((n))
            for i in range(1,n-1):
                ut[i] = (u[i+1]-u[i-1]) / (2*dt)

            ut[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dt
            ut[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dt
            return ut

        self.b = torch.enable_grad()(compute_ut)(rb, self.t_number, dt)

        self.loss = (self.F - self.b).norm(2)
        


    def freeze(self, is_freeze_test=True):
        for param in self.net_psi.parameters():
            param.requires_grad = not is_freeze_test
        for param in self.net_phi1.parameters():
            param.requires_grad = is_freeze_test
        for param in self.net_phi2.parameters():
            param.requires_grad = is_freeze_test

    def adversarial_update(self):
        for iter in range(self.step_neural):
            print(iter)
            self.optimizer.zero_grad()
            self.compute_loss()
            self.freeze(is_freeze_test=True)
            self.loss.backward()
            self.optimizer.step()
            for iter in range(self.step_test):
                self.freeze(is_freeze_test=False)
                self.optimizer.zero_grad()
                self.compute_loss()
                self.loss.backward()
                self.optimizer.step()
      
    @utils.timing
    def train(self, display_every=1):
        self.compute_loss()
        iter_run = 0
        while iter_run < self.max_iter:
            self.adversarial_update()
            if iter_run % display_every:
                print("Iteration: ", iter_run, "loss: ", self.loss)
            if self.loss < self.error_tolerance:
                break

if __name__ == '__main__':
    # t = np.loadtxt('./data/t.txt')
    # data = np.loadtxt('./data/sampling.txt')
    # t = np.random.randn(10).astype(np.float32)
    # data = np.random.randn(10,200,3).astype(np.float32)
    t = np.linspace(0,10,1001).astype(np.float32)
    data = scipy.io.loadmat('./data/MC-MutiDim.mat')['bb'].astype(np.float32)
    t = torch.tensor(t)
    data = torch.tensor(data)
    dimension = data.shape[-1]
    testFunc = DNN([3]+[50]*3+[1])
    vFunc = DNN([3]+[50]*3+[1])
    sFunc = DNN([3]+[50]*3+[dimension])

    model = Model(t, data, testFunc, vFunc, sFunc)
    model.compile(step_test=10, step_neural=10, error_tolerance = 1e-3, max_iter=1e5, optimizer="Adam")
    model.train()