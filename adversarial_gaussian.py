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

class Gaussian(torch.nn.Module): 
    def __init__(self):
        super(Gaussian, self).__init__()
        self.mu = torch.nn.Parameter(torch.ones(1))
        self.sigma = torch.nn.Parameter(torch.ones(1))
        self.mu = nn.init.uniform_(self.mu)
        self.sigma = nn.init.uniform_(self.sigma)

    def gaussB(self, x):
        func = 1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi))) * torch.exp(-0.5*(x-self.mu)**2/self.sigma**2)
        return func

    def gaussZero(self, x):
        func = 1
        for d in range(x.shape[2]):
            func = func * self.gaussB(x[:, :, d])
        return func

    def gaussFirst(self, x, g0):
        func = torch.zeros([x.shape[0], x.shape[1], x.shape[2]])
        for k in range(x.shape[2]):
            func[:, :, k] = -(x[:, :, k] - self.mu)/self.sigma**2 * g0
        return func

    def gaussSecond(self, x, g0):
        func = torch.zeros([x.shape[0], x.shape[1], x.shape[2], x.shape[2]])
        for k in range(x.shape[2]):
            for j in range(x.shape[2]):
                if k == j:
                    func[:, :, k, j] = (-1/self.sigma**2 + (-(x[:, :, k]-self.mu)/self.sigma**2)
                                    *(-(x[:, :, j]-self.mu)/self.sigma**2))*g0
                else:
                    func[:, :, k, j] = (-(x[:, :, k]-self.mu)/self.sigma**2)*(-(x[:, :, j]-self.mu)/self.sigma**2)*g0
        return func
    
    def forward(self, x, diff_order=0):
        g0 = self.gaussZero(x)
        if diff_order == 0:
            return g0
        elif diff_order == 1:
            return self.gaussFirst(x, g0)
        elif diff_order == 2:
            return self.gaussSecond(x, g0)
        else:
            raise RuntimeError("higher order derivatives of the gaussian has not bee implemented!")

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

        
        # TBD
        # self.batch_size = None 
        # self.train_state = TrainState()
        # self.losshistory = LossHistory()


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
    
    @utils.timing
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
        # Phi = self.net(TX)
        gauss0 = self.net(TX, diff_order=0)
        gauss1 = self.net(TX, diff_order=1)
        gauss2 = self.net(TX, diff_order=2)
        

        # print("self.basis_number", self.basis_number)
        # print("self.dimension", self.dimension)
        for kd in range(self.dimension):
                for jb in range(self.basis_number):
                    H = 1/self.bash_size * torch.sum(
                        gauss1[:, :, kd]
                         * 
                        self.basis[:, :, jb],
                        dim = 1
                        )
                    self.A[:, kd*self.dimension+jb] = H

        # compute A by F_lkj
        for ld in range(self.dimension):
            for kd in range(self.dimension):
                for jb in range(self.basis_number):
                    # print("ld", ld)
                    # print("kd", kd)
                    # print("jb", jb)
                    F = 1/self.bash_size * torch.sum(
                        gauss2[:, :, ld, kd]
                         * 
                        self.basis[:, :, jb],
                        dim = 1
                        )
                    self.A[:, H_number+ld*self.dimension+kd*self.dimension+jb] = F
        rb = 1/self.bash_size * torch.sum(gauss0, dim=1).squeeze()
        dt = (torch.max(self.t)-torch.min(self.t)) / self.t_number
        def compute_ut(u, n, dt):   
            ut = torch.zeros((n))
            for i in range(1,n-1):
                ut[i] = (u[i+1]-u[i-1]) / (2*dt)

            ut[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dt
            ut[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dt
            return ut

        self.b = torch.enable_grad()(compute_ut)(rb, self.t_number, dt)
        
    @torch.no_grad()
    @utils.timing
    def solve_linear_regression(self):
        self.zeta = torch.tensor(np.linalg.lstsq(self.A.detach().numpy(), self.b.detach().numpy())[0])
        # TBD sparse regression
                
    @utils.timing
    def compile(self, step_adversarial = 10, error_tolerance = 1e-3, max_iter=1e5, optimizer="SGD", basis_order=3):  
        self.optimizer = utils.optimizer_get(self.net.parameters(), optimizer, learning_rate=1e-3)
        self.optimizer.param_groups[0]['lr'] *= -1
        self.error_tolerance = error_tolerance
        self.max_iter = max_iter
        self.step_adversarial = step_adversarial
        self.dimension = self.data.shape[-1]
        self.basis_order = basis_order
        self.build_basis()

    @utils.timing  
    def adversarial_update(self):
        for iter in range(self.step_adversarial):
            print(iter)
            self.optimizer.zero_grad()
            self.build_linear_system() # 1.5s 
            loss = (torch.matmul(self.A, self.zeta)-self.b).norm(2)  
            self.loss = loss
            # loss.backward() # 14s 
            # self.optimizer.step()
      
    @utils.timing
    def train(self, display_every=1):
        self.build_linear_system()
        self.solve_linear_regression()
        iter_run = 0
        while iter_run < self.max_iter:
            self.adversarial_update()
            if iter_run % display_every==0:
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
    t = np.linspace(0,20,1001).astype(np.float32)
    data = scipy.io.loadmat('./data/data.mat')['bb'].astype(np.float32)
    t = torch.tensor(t)
    data = torch.tensor(data).unsqueeze(-1)
    # testFunc = DNN([3]+[50]*3+[1])
    testFunc = Gaussian()

    model = Model(t, data, testFunc)
    model.compile(step_adversarial = 1, error_tolerance = 1e-3, max_iter=1e5, optimizer="Adam")
    model.train()