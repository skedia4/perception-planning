#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@Time    :   2022/05/06 18:14:18
@Author  :   Yu Zhou 
@Contact :   yuzhou7@illinois.edu
'''

import numpy as np
from planning.misc import *
from planning.nmpc import *
from planning.ca_func import *
from planning.edt_map import *

class nmpcDiffDrive(nmpc):
    def __init__(self, sys, edt):
        super().__init__(sys)

        opts = {}
        opts["enable_fd"]=True # enable finite differencing
        opts["enable_forward"]=False # disable forward mode AD
        opts["enable_reverse"]=False # disable reverse mode AD
        opts["enable_jacobian"]=False # disable AD by calculating full Jacobian
        opts["fd_method"] = "central" # specify FD scheme: analytical,forward, central, backward, smoothing
        opts["fd_options"] = {'h_iter': False, 'h': 1e-2}
        self.f = edtCostCb('f', d = 2, edt = edt, opts = opts) 

        self.lambda_tracking, self.lambda_edt, self.lambda_terminal, self.lambda_energy \
        = sys.optParam['lambdas'][0], sys.optParam['lambdas'][1], sys.optParam['lambdas'][2], sys.optParam['lambdas'][3]
        self.obj_tracking = self.getTTCost() * self.lambda_tracking
        self.obj_edt = self.getEDTCost() * self.lambda_edt
        self.obj_terminal = self.getTerminalCost() * self.lambda_terminal
        self.obj_energy = self.getEnergyCost() * self.lambda_energy
        self.cost = self.obj_tracking + self.obj_edt + self.obj_energy + self.obj_terminal
        self.opti.minimize(self.cost)

    def getTTCost(self):
        obj = 0
        for i in range(self.N-1):
            obj +=  ca.mtimes([(self.opt_states[i, :]-self.opt_xs.T), self.Q, (self.opt_states[i, :]-self.opt_xs.T).T])
        return obj
    
    def getTerminalCost(self):
        obj = (ca.mtimes([(self.opt_states[self.N-1, :]-self.opt_xs.T), self.Q_N, (self.opt_states[self.N-1, :]-self.opt_xs.T).T]))
        return obj

    def getEnergyCost(self):
        obj = 0
        for i in range(self.N-1):
            # obj +=  (ca.mtimes([(self.opt_controls[i+1, :] - self.opt_controls[i, :]), self.R, (self.opt_controls[i+1, :] - self.opt_controls[i, :]).T]))   # smooth control
            obj +=  ca.mtimes([self.opt_controls[i, :], self.R, self.opt_controls[i, :].T])
        return obj

    def getEDTCost(self):
        cost = 0
        for i in range(1, self.N):
            # note x, y is reversed when evaluating with EDT
            cost += self.f(self.opt_states[i, 1], self.opt_states[i, 0]) #[-y, x], -y for simulator, y for occ image
        return cost

class DiffDriveSys():
    def __init__(self, cfg, edt, seed = 0):
        print("Init differential drive Model")
        self.sets = cfg['space_sets']
        self.dt = cfg['system']['dt']
        self.ctrl_dim = cfg['system']['ctrl_dim']
        self.obs_dim = cfg['system']['obs_dim']
        self.name = cfg['system']['name']
        self.optParam = cfg['trajOpt']
        self.opt = nmpcDiffDrive(self, edt)

    def model(self, x_, u_):
        """
        dynamics in casadi format
        @return: lambda function
        """
        x_dot = self.dynamics(x_, u_)
        return ca.horzcat(x_dot[0], x_dot[1], x_dot[2])

    def dynamics(self, x, u):
        """
        Dynamics of cartpole system
        @param y: array, states
        @param u: array, control

        @Return
            A list describing the dynamics of the cartpole
        """
        return [u[0]*np.cos(x[2]),
                u[0]*np.sin(x[2]),
                u[1]]

    def next(self, y, u):
        """
        Forward simulation with dynamics 
        @param y: array, states
        @param u: array, control
        @param reverse: bool, True for viable set and False for reachable set
        @return y: array, next states given previous state and control input
        """
        y += self.dt * np.asarray(self.dynamics(y, u))
        y[2] =  diff_angle(y[2],0)
        return y

    def predict(self, state, u, n, dt = 0.02):
        for i in range(n):
            state += np.asarray(self.dynamics(state, u))*dt
            state[2] = diff_angle(state[2],0)
        return np.around(state,4)
        