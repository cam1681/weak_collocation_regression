"""
Function: data-driven reveal SDE by the weak form of FKE: 1. linear regression 2. adversarial update
@author pi square 
@email: hpp1681@gmail.com
created in Oct 11, 2021
update log:
    0. Oct 11, 2021: created
    1. Oct 15, 2021: change time index loop to Tensor form including the time index as the first dimension
    2. Oct 28, 2021: fix bugs of self.t_number -> self.bash_size
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
    def __init__(self, layers = [3]+[10]*4+[1]):
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
            layer_list.append(('activation_%d' % i, self.relu()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
            )
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
    def __init__(self, t, data, testFunc):
        self.t = t
        self.itmax = len(t)
        self.data = data
        self.net = testFunc
        self.basis = None # given by build_basis
        self.A = None # given by build_A
        self.b = None # given by build_b
        self.dimension = None
        self.basis_number = None
        self.basis_order = None
        self.bash_size = data.shape[1]

        self.zeta = None # coefficients of the unknown function
        self.error_tolerance = None
        self.max_iter = None
        self.loss = None

        
        # self.batch_size = None # tbd
        # self.train_state = TrainState() # tbd
        # self.losshistory = LossHistory() # tbd


    def _get_data_t(self, it):
        X = self.data[it,:,:]
        return X
    
    @utils.timing # decorator
    @torch.no_grad()
    def build_basis(self): # \Lambda matrix
        """build the basis list for the different time snapshot 
        """
        self.t_number = len(self.t)
        self.basis_number = int(np.math.factorial(self.dimension+self.basis_order)
                /(np.math.factorial(self.dimension)*np.math.factorial(self.basis_order)))
        self.basis = []

        for it in range(self.t_number):
            X = self._get_data_t(it)
            basis_count = 0
            Theta = torch.zeros(X.size(0),self.basis_number)
            Theta[:,0] = 1
            basis_count += 1
            for ii in range(0,self.dimension):
                Theta[:,basis_count] = X[:,ii]
                basis_count += 1

            if self.basis_order >= 2:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        Theta[:,basis_count] = torch.mul(X[:,ii],X[:,jj])
                        basis_count += 1

            if self.basis_order >= 3:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        for kk in range(jj,self.dimension):
                            Theta[:,basis_count] = torch.mul(torch.mul(X[:,ii],
                                X[:,jj]),X[:,kk])
                            basis_count += 1

            if self.basis_order >= 4:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        for kk in range(jj,self.dimension):
                            for ll in range(kk,self.dimension):
                                Theta[:,basis_count] = torch.mul(torch.mul(torch.mul(X[:,ii],
                                    X[:,jj]),X[:,kk]),X[:,ll])
                                basis_count += 1

            if self.basis_order >= 5:
                for ii in range(0,self.dimension):
                    for jj in range(ii,self.dimension):
                        for kk in range(jj,self.dimension):
                            for ll in range(kk,self.dimension):
                                for mm in range(ll,self.dimension):
                                    Theta[:,basis_count] = torch.mul(torch.mul(torch.mul(torch.mul(
                                        X[:,ii],X[:,jj]),X[:,kk]),
                                            X[:,ll]),X[:,mm])
                                    basis_count += 1
            assert basis_count == self.basis_number
            self.basis.append(Theta)
        self.basis = torch.stack(self.basis)
    
    
    def build_linear_system(self):
        H_number = self.dimension * self.basis_number
        F_number = self.dimension * self.dimension * self.basis_number
        
        self.A = torch.zeros([self.t_number, H_number+F_number])
        rb = torch.zeros(self.t_number)
        self.b = torch.zeros(self.t_number)


        # ##########################################################
        #  Tensor form of computing A and b for parallel computing
        # ##########################################################

        TX = self.data
        TX.requires_grad = True
        Phi = self.net(TX)
        

        for kd in range(self.dimension):
                
                for jb in range(self.basis_number):
                    H = 1/self.bash_size * torch.sum(
                        torch.autograd.grad(
                            outputs=Phi, inputs=TX, 
                            grad_outputs=torch.ones_like(Phi),
                            create_graph=True,
                            retain_graph=True)[0][:,:,kd] * 
                        self.basis[:, :, jb],
                        dim = 1
                        )
                    self.A[:, kd*self.dimension+jb] = H

        # compute A by F_lkj
        for ld in range(self.dimension):
            for kd in range(self.dimension):
                for jb in range(self.basis_number):
                    
                    
                    output=torch.autograd.grad(
                                outputs=Phi, inputs=TX, 
                                grad_outputs=torch.ones_like(Phi),
                                create_graph=True,
                                retain_graph=True
                                )[0][:,:,ld]
                    F = 1/self.bash_size * torch.sum(
                        torch.autograd.grad(
                            outputs=output,
                            inputs=TX,
                            grad_outputs=torch.ones_like(output),
                            create_graph=True,
                            retain_graph=True
                            )[0][:,:,kd] * 
                        self.basis[:, :, jb],
                        dim = 1
                        )
                    self.A[:, H_number+ld*self.dimension+kd*self.dimension+jb] = -F
        rb = 1/self.bash_size * torch.sum(Phi, dim=1).squeeze()
        dt = (torch.max(self.t)-torch.min(self.t)) / self.t_number
        def compute_ut(u, n, dt):   
            ut = torch.zeros((n))
            for i in range(1,n-1):
                ut[i] = (u[i+1]-u[i-1]) / (2*dt)

            ut[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dt
            ut[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dt
            return ut

        self.b = torch.enable_grad()(compute_ut)(rb, self.t_number, dt)
        

        ## Prolbem arises? How to pass the loss lose through the fitting?
        # self.b = torch.enable_grad()(utils.compute_b)(rb, dt, time_diff = 'poly', 
        # lam_t = None, width_t = None, deg_t = None, sigma = 2) 

        # ##########################################################
        #  forloop of computing A and b for time t
        # ##########################################################
        # for it in range(self.t_number): 
        #     X = self._get_data_t(it)
        #     X.requires_grad = True
        #     phi = self.net(X)
            
        #     rb[it] = 1/self.t_number * torch.sum(phi)
        #     # compute A by H_kj
        #     for kd in range(self.dimension):
        #         for jb in range(self.basis_number):
        #             H = 1/self.t_number * torch.dot(
        #                 torch.autograd.grad(
        #                     outputs=phi, inputs=X, 
        #                     grad_outputs=torch.ones_like(phi),
        #                     create_graph=False,
        #                     retain_graph=True)[0][:,kd],
        #                 self.basis[it][:, jb]
        #                 )
        #             self.A[it, kd*self.dimension+jb] = H

        #     # compute A by F_lkj
        #     for ld in range(self.dimension):
        #         for kd in range(self.dimension):
        #             for jb in range(self.basis_number):
        #                 output=torch.autograd.grad(
        #                             outputs=phi, inputs=X, 
        #                             grad_outputs=torch.ones_like(phi),
        #                             create_graph=True,
        #                             retain_graph=True
        #                             )[0][:,ld]
        #                 F = 1/self.t_number * torch.dot(
        #                     torch.autograd.grad(
        #                         outputs=output,
        #                         inputs=X,
        #                         grad_outputs=torch.ones_like(output),
        #                         create_graph=False,
        #                         retain_graph=True
        #                         )[0][:,kd],
        #                     self.basis[it][:, jb]
        #                     )
        #                 self.A[it, H_number+ld*self.dimension+kd*self.dimension+jb] = -F
        # dt = self.t / self.t_number
        # self.b = utils.compute_b(rb, dt, time_diff = 'poly', 
        #     lam_t = None, width_t = None, deg_t = None, sigma = 2)

    
    @torch.no_grad()
    def solve_linear_regression(self):
        self.zeta = torch.tensor(np.linalg.lstsq(self.A.detach().numpy(), self.b.detach().numpy())[0])
    
        # tbd sparse regression
                
    @utils.timing
    def compile(self, step_adversarial = 10, error_tolerance = 1e-3, max_iter=1e5, optimizer="SGD", basis_order=3):  
        self.optimizer = utils.optimizer_get(self.net.parameters(), optimizer, learning_rate=-1e-3)
        self.error_tolerance = error_tolerance
        self.max_iter = max_iter
        self.step_adversarial = step_adversarial
        self.dimension = self.data.shape[-1]
        self.basis_order = basis_order
        self.build_basis()
        
    def adversarial_update(self):
        for iter in range(self.step_adversarial):
            self.optimizer.zero_grad()
            self.build_linear_system()
            loss = (torch.matmul(self.A, self.zeta)-self.b).norm(2)  
            self.loss = loss
            loss.backward()
            self.optimizer.step()
      
    @utils.timing
    def train(self, display_every=1):
        self.build_linear_system()
        self.solve_linear_regression()
        iter_run = 0
        while iter_run < self.max_iter:
            self.adversarial_update()
            if iter_run % display_every == 0:
                print("Iteration: ", iter_run, "loss: ", self.loss)
            if self.loss < self.error_tolerance:
                break
            self.solve_linear_regression()
            print(self.zeta)

        

if __name__ == '__main__':
    # t = np.loadtxt('./data/t.txt')
    # data = np.loadtxt('./data/sampling.txt')
    # t = np.random.randn(10).astype(np.float32)
    # data = np.random.randn(10,200,3).astype(np.float32)
    t = np.linspace(0,10,1001).astype(np.float32)
    data = scipy.io.loadmat('./data/MC-MutiDim.mat')['bb'].astype(np.float32)
    t = torch.tensor(t)
    data = torch.tensor(data)
    testFunc = DNN([3]+[10]*2+[1])
    model = Model(t, data, testFunc)
    model.compile(step_adversarial = 1, error_tolerance = 1e-3, max_iter=1e5, optimizer="Adam")
    model.train()