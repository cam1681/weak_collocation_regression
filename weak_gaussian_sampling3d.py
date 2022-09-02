"""
Function: data-driven reveal SDE by the weak form of FKE: 1. linear regression 2. adversarial update
@author pi square 
@email: hpp1681@gmail.com
created in Oct 11, 2021
update log:
    0. Oct 11, 2021: created
    1. Oct 15, 2021: change time index loop to Tensor form including the time index as the first dimension
    2. Oct 28, 2021: fix bugs of self.t_number -> self.bash_size
    3. Dec 9, 2021: modifies the adversarial.py to weak_gaussian_sampling for sampling in the test function
    4. Apri 10, 2022: deal with 3d data
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import time
import utils
import scipy.io
from pyDOE import lhs

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


device = torch.device("cpu")
# torch.manual_seed(1234)
# np.random.seed(1234)


# TBD: extend the dimension
class Gaussian(torch.nn.Module): 
    def __init__(self, mu, sigma):
        super(Gaussian, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.dim = mu.shape[0]

    def gaussZero(self, x):
        func = 1
        for d in range(self.dim):
            func = func * 1/(self.sigma*torch.sqrt(2*torch.tensor(torch.pi))) * torch.exp(-0.5*(x[:,:,d]-self.mu[d])**2/self.sigma**2)
        return func

    def gaussFirst(self, x, g0):
        func = torch.zeros([x.shape[0], x.shape[1], x.shape[2]])
        for k in range(self.dim):
            func[:, :, k] = -(x[:, :, k] - self.mu[k])/self.sigma**2 * g0
        return func

    def gaussSecond(self, x, g0):
        func = torch.zeros([x.shape[0], x.shape[1], x.shape[2], x.shape[2]])
        for k in range(x.shape[2]):
            for j in range(x.shape[2]):
                if k == j:
                    func[:, :, k, j] =  (
                                    -1/self.sigma**2 + (-(x[:, :, k]-self.mu[k])/self.sigma**2)
                                    * (-(x[:, :, j]-self.mu[j])/self.sigma**2)
                                    ) * g0
                else:
                    func[:, :, k, j] =  (-(x[:, :, k]-self.mu[k])/self.sigma**2)*(
                        -(x[:, :, j]-self.mu[j])/self.sigma**2
                        ) * g0
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
        basis = []

        for it in range(self.t_number):
            X = self._get_data_t(it)
            basis_count = 0
            Theta = torch.zeros(X.size(0),self.basis_number)
            Theta[:,0] = 1
            basis_count += 1

            if self.basis_order >= 1:
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
            basis.append(Theta)
            # print("X", X)
            # print("theta", Theta)

        self.basis = torch.stack(basis)
        # print("self.basis.shape", self.basis.shape)
    
    def computeLoss(self):
        return (torch.matmul(self.A, torch.tensor(self.zeta).to(torch.float).unsqueeze(-1))-self.b.unsqueeze(-1)).norm(2) 

    def computeTrueLoss(self):
        return (torch.matmul(self.A, self.zeta_true)-self.b.unsqueeze(-1)).norm(2) 

    def computeAb(self, gauss):
        H_number = self.dimension * self.basis_number
        F_number = self.dimension * self.dimension * self.basis_number
        
        A = torch.zeros([self.t_number, H_number+F_number])
        rb = torch.zeros(self.t_number)
        b = torch.zeros(self.t_number)


        # ##########################################################
        #  Tensor form of computing A and b for parallel computing
        # ##########################################################

        TX = self.data
        TX.requires_grad = True
        # Phi = self.net(TX)
        gauss0 = gauss(TX, diff_order=0)
        gauss1 = gauss(TX, diff_order=1)
        gauss2 = gauss(TX, diff_order=2)
        

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
                    A[:, kd*self.basis_number+jb] = H

        # compute A by F_lkj
        for ld in range(self.dimension):
            for kd in range(self.dimension):
                for jb in range(self.basis_number):
                    F = 1/self.bash_size * torch.sum(
                        gauss2[:, :, ld, kd]
                         * 
                        self.basis[:, :, jb],
                        dim = 1
                        )
                    A[:, H_number+ld*self.dimension*self.basis_number+kd*self.basis_number+jb] = F
        rb = 1/self.bash_size * torch.sum(gauss0, dim=1).squeeze()
        dt = (torch.max(self.t)-torch.min(self.t)) / self.t_number

        b = torch.tensor(torch.enable_grad()(utils.compute_b)(rb, dt, time_diff = 'Tik'))
        # print("b.shape", b.shape)

        # plot
        # plt.clf()
        # plt.plot(rb.detach().numpy(),'-*')
        # plt.plot(b.detach().numpy(),'-o')
        # plt.draw()
        # plt.pause(1)
     
        # print("b", b)
        # print("A.shape", A.shape)
        return A, b
    
    
    def sampleTestFunc(self, samp_number):
        # for i in range(self.sampling_number):
        lb = torch.tensor([self.data[:,:,i].min() for i in range(self.dimension)]).to(device)
        ub = torch.tensor([self.data[:,:,i].max() for i in range(self.dimension)]).to(device)
        mu_list = lb + (ub-lb)*torch.tensor(lhs(self.dimension, samp_number)).to(device)
        # mu_list = torch.rand(samp_number)*(self.data.max()-self.data.min()) + self.data.min()
        print("mu_list", mu_list)
        sigma_list = torch.ones(samp_number)
        # print("mu_list", mu_list)
        # print("sigma_list", sigma_list)
        return mu_list, sigma_list

    def buildLinearSystem(self, samp_number):
        mu_list, sigma_list = self.sampleTestFunc(samp_number)
        A_list = []
        b_list = []
        for i in range(mu_list.shape[0]): #
            mu = mu_list[i]
            sigma = sigma_list[i]
            gauss = self.net(mu, sigma)
            A, b = self.computeAb(gauss)
            A_list.append(A)
            b_list.append(b)
        # print("A_list", A_list)
        # print("b_list", b_list)
        self.A = torch.cat(A_list, dim=0) # 2-dimension
        self.b = torch.cat(b_list, dim=0).unsqueeze(-1) # 1-dimension


    @torch.no_grad()
    @utils.timing
    def solveLinearRegress(self):
        self.zeta = torch.tensor(np.linalg.lstsq(self.A.detach().numpy(), self.b.detach().numpy())[0])
        # TBD sparse regression
    
    
    def STRidge(self, X0, y, lam, maxit, tol, normalize = 0, print_results = False):
        """
        Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
        approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

        This assumes y is only one column
        """
        n,d = X0.shape
        X = np.zeros((n,d), dtype=np.complex64)
        # First normalize data
        if normalize != 0:
            Mreg = np.zeros((d,1))
            for i in range(0,d):
                Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
                X[:,i] = Mreg[i]*X0[:,i]
        else: X = X0

        # Get the standard ridge esitmate
        if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y))[0]
        else: w = np.linalg.lstsq(X,y)[0]
        num_relevant = d
        biginds = np.where(abs(w) > tol)[0]

        # Threshold and continue
        for j in range(maxit):
            # Figure out which items to cut out
            smallinds = (np.where((abs(w) < tol)) or np.where(abs(w) > 1e8))[0] # extend to big inds by pi
            print("smallinds", smallinds)
            new_biginds = [i for i in range(d) if i not in smallinds]

            # If nothing changes then stop
            if num_relevant == len(new_biginds): break
            else: num_relevant = len(new_biginds)

            # Also make sure we didn't just lose all the coefficients
            if len(new_biginds) == 0:
                if j == 0:
                    #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                    return w
                else: break
            biginds = new_biginds

            # Otherwise get a new guess
            w[smallinds] = 0
            if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
            else: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

        # Now that we have the sparsity pattern, use standard least squares to get w
        if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

        if normalize != 0: return np.multiply(Mreg,w)
        else: return w
    
    @utils.timing
    def compile(self, error_tolerance = 1e-3, max_iter=1e5, basis_order=1, max_sgd=100, lr=0.001, lb=0.001):  
        self.error_tolerance = error_tolerance
        self.max_iter = max_iter
        self.max_sgd = max_sgd
        self.lr = lr
        self.lb = lb
        self.dimension = self.data.shape[-1]
        self.basis_order = basis_order
        self.build_basis()
        print("basis_order: ", self.basis_order)
        print("basis_number: ", self.basis_number)

    @utils.timing
    @torch.no_grad()
    def train(self, display_every=1):
        self.buildLinearSystem(samp_number=500)
        # print("self.A.shape", self.A.shape)
        # print("self.b.shape", self.b.shape)
        self.zeta = torch.tensor(self.STRidge(self.A.detach().numpy(), self.b.detach().numpy(), 0.001, 100, 1e-1)).to(torch.float)
        print("self.zeta", self.zeta)

        # self.solveLinearRegress()
        # for iter in range(self.max_sgd):
        #     print("iter", iter)
        #     self.buildLinearSystem(samp_number=20)
        #     d_zeta = self.A.t().matmul(self.A.matmul(self.zeta)-self.b) + self.lb*torch.sign(self.zeta)
        #     print("d_zeta", d_zeta)
        #     self.zeta = self.zeta - self.lr * d_zeta
        #     # self.zeta = torch.tensor(self.STRidge(self.A.detach().numpy(), self.b.detach().numpy(), 0.1, 100, 1e-3)).to(torch.float)
        #     print("self.zeta: \n", self.zeta)

        # print("self.zeta", self.zeta)
        # print("self.A", self.A)
        # print("self.b", self.b)
        

        # self.zeta_true = torch.tensor([0.5, -1, 0, 0.5, 0, 0]).unsqueeze(-1)
        # # self.zeta_true = torch.tensor(self.zeta).to(torch.float)
        # # self.zeta_true = torch.tensor([ 0.49023819, -1.26118481,  0.18142581,  0.61106366, -1.03349543,  0.6275996 ]
        # #                     ).unsqueeze(-1)
        # print("self.zeta", torch.tensor(self.zeta).to(torch.float).unsqueeze(-1))
        # print("self.zeta_true", self.zeta_true)
        # print("self.zeta-self.zeta_true", torch.tensor(self.zeta).to(torch.float).unsqueeze(-1)-self.zeta_true)
        

if __name__ == '__main__':
    # t = np.loadtxt('./data/t.txt')
    # data = np.loadtxt('./data/sampling.txt')
    # t = np.random.randn(3).astype(np.float32)
    # data = np.random.randn(3,2,1).astype(np.float32)
    # t = np.array([0, 1, 2]).astype(np.float32)
    # data = np.array([[[2,2],[3,3]], [[2,2],[3,3]],[[4,4],[5,5]]]).astype(np.float32)
    t = np.linspace(0,10,501).astype(np.float32)
    data = scipy.io.loadmat('./data/data3d.mat')['bb'].astype(np.float32)
    t = torch.tensor(t)
    # data = torch.tensor(data).unsqueeze(-1)
    # data = torch.tensor(data)[:,:,0].unsqueeze(-1)
    data = torch.tensor(data)
    print("data.shape", data.shape)
    # testFunc = DNN([3]+[50]*3+[1])
    testFunc = Gaussian

    model = Model(t, data, testFunc)
    model.compile(error_tolerance=1e-3, max_iter=1e5, basis_order=1, max_sgd=1000, lr=0.001, lb=0.00)
    model.train()
    # print("loss: ", model.computeLoss())
    # print("True loss: ", model.computeTrueLoss())