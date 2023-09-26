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
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
# from data.GenerateData_inequal import DataSet
from data.GenerateData_fun import DataSet
import time
import utils
import scipy.io

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Gaussian(torch.nn.Module): 
    def __init__(self, mu, sigma):
        super(Gaussian, self).__init__()
        self.mu = mu
        self.sigma = sigma

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
                    func[:, :, k, j] =  (
                                    -1/self.sigma**2 + (-(x[:, :, k]-self.mu)/self.sigma**2)
                                    * (-(x[:, :, j]-self.mu)/self.sigma**2)
                                    ) * g0
                else:
                    func[:, :, k, j] =  (-(x[:, :, k]-self.mu)/self.sigma**2)*(
                        -(x[:, :, j]-self.mu)/self.sigma**2
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
            for i in range(self.basis_number):
                Theta[:, i] = X[:, 0]**i
            basis.append(Theta)
            # print("X", X)
            # print("theta", Theta.shape)

        self.basis = torch.stack(basis)
        print("self.basis.shape", self.basis.shape)
    
    def computeLoss(self):
        return (torch.matmul(self.A, torch.tensor(self.zeta).to(torch.float).unsqueeze(-1))-self.b.unsqueeze(-1)).norm(2) 

    def computeTrueLoss(self):
        return (torch.matmul(self.A, self.zeta_true)-self.b.unsqueeze(-1)).norm(2) 

    def computeAb(self, gauss):
        H_number = self.dimension * self.basis_number
        F_number = self.dimension * self.dimension #* self.basis_number
        
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
                # print("gauss1[:, :, %s]" % kd, gauss1[:, :, kd].size())
                H = 1/self.bash_size * torch.sum(
                    gauss1[:, :, kd]
                     *
                    self.basis[:, :, jb], dim=1
                    )
                A[:, kd*self.basis_number+jb] = H

        # compute A by F_lkj
        for ld in range(self.dimension):
            for kd in range(self.dimension):
                F = 1/self.bash_size * torch.sum(
                    gauss2[:, :, ld, kd], dim=1
                    )
                A[:, H_number] = F
        rb = 1/self.bash_size * torch.sum(gauss0, dim=1).squeeze()
        dt = (torch.max(self.t)-torch.min(self.t)) / (self.t_number - 1)
        # print("b", rb)

        # b = torch.tensor(torch.enable_grad()(utils.compute_b)(rb, dt, time_diff='Tik'))
        # print("b.shape", b.shape)

        # plt.clf()
        # plt.plot(rb.detach().numpy(),'-*')
        # plt.plot(b.detach().numpy(),'-o')
        # plt.draw()
        # plt.pause(1)

        # print("b", b)
        # print("A.shape", A.shape)
        if self.type == 'PDEFind':
            b = torch.tensor(torch.enable_grad()(utils.compute_b)(rb, dt, time_diff='Tik'))
            return A, b
        if self.type == 'LMM_2':
            AA = torch.ones(A.size(0) - 1, A.size(1))
            for i in range(AA.size(0)):
                AA[i, :] = (A[i, :] + A[i + 1, :]) / 2
            bb = torch.from_numpy(np.diff(rb.numpy())) / dt
            return AA, bb
        if self.type == 'LMM_3':
            AA = torch.ones(A.size(0) - 2, A.size(1))
            bb = torch.ones(A.size(0) - 2)
            for i in range(AA.size(0)):
                AA[i, :] = (A[i, :] + 4*A[i + 1, :] + A[i + 2, :]) * dt / 3
                bb[i] = rb[i + 2] - rb[i]
            return AA, bb
        
        if self.type == 'LMM_6':
            AA = torch.ones(A.size(0) - 5, A.size(1))
            bb = torch.ones(A.size(0) - 5)
            for i in range(AA.size(0)):
                AA[i, :] = (
                        A[i+1, :] +
                        1/2 * (A[i+2, :] + A[i+1, :]) +
                        5/12 * A[i+3, :] + 8/12 * A[i+2, :] - 1/12 * A[i+1, :] +
                        9/24 * A[i+4, :] + 19/24 * A[i+3, :] - 5/24 * A[i+2, :] + 1/24 * A[i+1, :] +
                        251/720 * A[i + 5, :] + 646/720 * A[i + 4, :] - 264/720 * A[i + 3, :] + 106/720 * A[i + 2, :] - 19/720 * A[i + 1, :]
                            ) * dt
                bb[i] = rb[i + 5] - rb[i]
            return AA, bb  

        if self.type == 'bdf2':
            AA = torch.ones(A.size(0) - 2, A.size(1))
            bb = torch.ones(A.size(0) - 2)
            for i in range(AA.size(0)):
                AA[i, :] = (A[i, :] + 4*A[i + 1, :] + A[i + 2, :]) * dt / 3
                bb[i] = rb[i + 2] - rb[i]
            return AA, bb  
        if self.type == 'LMM_2_nonequal':
            AA = torch.ones(A.size(0) - 1, A.size(1))
            bb = torch.ones(A.size(0) - 1)
            ht = torch.from_numpy(np.diff(self.t.numpy()))
            for i in range(AA.size(0)):
                AA[i, :] = (A[i, :] + A[i + 1, :]) / 2 * ht[i]
                bb[i] = rb[i + 1] - rb[i]
            return AA, bb
        if self.type == 'non-equal3':
            AA = torch.ones(A.size(0) - 2, A.size(1))
            bb = torch.ones(A.size(0) - 2)
            ht = torch.from_numpy(np.diff(self.t.numpy()))
            # print("ht: ", ht)
            wt = torch.tensor([ht[i + 1] / ht[i] for i in range(ht.size(0) - 1)])
            # print("wt: ", wt)
            for i in range(AA.size(0)):
                print("ht[i + 1]", ht[i + 1], "wt[i]", wt[i])
                AA[i, :] = ht[i + 1] * (1 + wt[i]) / (1 + 2 * wt[i]) * A[i + 2, :]
                bb[i] = rb[i + 2] - (1 + wt[i]) ** 2 / (1 + 2 * wt[i]) * rb[i + 1] + wt[i] ** 2 / (1 + 2 * wt[i]) * rb[i]
            return AA, bb
        if self.type == 'non-equal-adams':
            AA = torch.ones(A.size(0) - 2, A.size(1))
            bb = torch.ones(A.size(0) - 2)
            ht = torch.from_numpy(np.diff(self.t.numpy()))
            # print("ht: ", ht)
            wt = torch.tensor([ht[i + 1] / ht[i] for i in range(ht.size(0) - 1)])
            # print("wt: ", wt)
            for i in range(AA.size(0)):
                AA[i, :] = ht[i + 1] / (6 * (1 + wt[i])) * (
                        (3 + 2 * wt[i]) * A[i + 2, :] 
                        + (3 + wt[i]) * (1 + wt[i]) * A[i + 1, :]
                        - wt[i] ** 2 * A[i, :])
                bb[i] = rb[i + 2] - rb[i + 1]
            return AA, bb
    def sampleTestFunc(self, samp_number):
        # for i in range(self.sampling_number):
        if self.gauss_samp_way == 'lhs':
            mu_list = self.lhs_ratio * torch.rand(samp_number)*(self.data.max()-self.data.min()) + self.data.min()
        if self.gauss_samp_way == 'SDE':
            if samp_number <= self.bash_size:
                index = np.arange(self.bash_size)
                np.random.shuffle(index)
                mu_list = data[-1, index[0: samp_number], :]
        # print("mu_list", mu_list)
        sigma_list = torch.ones(samp_number)*self.variance
        return mu_list, sigma_list

    def buildLinearSystem(self, samp_number):
        mu_list, sigma_list = self.sampleTestFunc(samp_number)
        A_list = []
        b_list = []
        for i in range(mu_list.shape[0]):
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

    @utils.timing
    def solveLinearRegress(self):
        self.zeta = torch.tensor(np.linalg.lstsq(self.A.detach().numpy(), self.b.detach().numpy())[0])
        # TBD sparse regression

    @utils.timing
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
            smallinds = np.where(abs(w) < tol)[0]
            print("STRidge_j: ", j)
            print("smallinds", smallinds)
            new_biginds = [i for i in range(d) if i not in smallinds]

            # If nothing changes then stop
            if num_relevant == len(new_biginds):
                print("here1")
                break
            else: num_relevant = len(new_biginds)

            # Also make sure we didn't just lose all the coefficients
            if len(new_biginds) == 0:
                if j == 0:
                    print("here2")
                    #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                    return w
                else:
                    print("here3")
                    break
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
    def compile(self, basis_order, gauss_variance, type, drift_term, gauss_samp_way, lhs_ratio):
        self.dimension = self.data.shape[-1]
        self.basis_order = basis_order
        self.build_basis()
        self.variance = gauss_variance
        self.type = type
        self.drift = drift_term
        self.gauss_samp_way = gauss_samp_way
        self.lhs_ratio = lhs_ratio if self.gauss_samp_way == 'lhs' else 1

    def plot_drift(self, interval=[-2, 2]):
        x = torch.linspace(interval[0], interval[1], 2500)
        y_true = self.drift(x)
        y_poly = torch.zeros_like(x)
        for i in range(self.basis_order+1):
            y_poly = y_poly + self.zeta[i, 0] * x**i
        plt.figure()
        plt.plot(x, y_true, label="true")
        plt.plot(x, y_poly, label="basis")
        plt.legend()
        error = torch.sum((y_true-y_poly)**2).numpy() / torch.sum(y_true**2).numpy()
        plt.title("order=%d, L2error=%.4f" % (self.basis_order, error))
        plt.show()
        print("L2 relative error:", error)

    @utils.timing
    @torch.no_grad()
    def train(self, gauss_samp_number, lam, STRidge_threshold):
        self.buildLinearSystem(samp_number=gauss_samp_number)
        print("A: ", self.A.size(), "b: ", self.b.size())
        self.zeta = torch.tensor(self.STRidge(self.A.detach().numpy(), self.b.detach().numpy(), lam, 100, STRidge_threshold)).to(torch.float)
        pinv_norm = np.linalg.norm(np.linalg.pinv(self.A.detach().numpy()), ord=np.inf)
        print("zeta: ", self.zeta)
        self.zeta[-1] = torch.sqrt(self.zeta[-1]*2)
        print("Diffusion term: ", self.zeta[-1])
        self.plot_drift(interval=[-1, 1])


if __name__ == '__main__':
    np.random.seed(100)
    torch.manual_seed(100)

    dt = 0.001
    t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    def drift(x):
        # return x - x**3
        return -2*x*np.exp(-x**2)
    def diffusion(x):
        return 1
    samples = 10000
    dim = 1
    dataset = DataSet(t, dt=dt, samples_num=samples, dim=1, drift_fun=drift, diffusion_fun=diffusion,
                      initialization=torch.normal(mean=0., std=0.2, size=(samples, 1)), explosion_prevention=False)
    data = dataset.get_data(plot_hist=False)
    print("data: ", data.shape, data.max(), data.min())

    testFunc = Gaussian
    model = Model(t, data, testFunc)
    model.compile(basis_order=9, gauss_variance=1, type='LMM_2_nonequal', drift_term=drift,
                  gauss_samp_way='lhs', lhs_ratio=1)
    model.train(gauss_samp_number=20, lam=0.0, STRidge_threshold=0.)
