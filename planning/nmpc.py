#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@Time    :   2022/05/06 18:14:43
@Author  :   Yu Zhou 
@Contact :   yuzhou7@illinois.edu
'''

import numpy as np
import casadi as ca

class nmpc:
    """
    Todo: seperate nmpc with objective funtion
    nonlinear mpc
    """
    def __init__(self, sys, mode = 'TT', N = None):
        self.sys = sys
        self.opti = ca.Opti()
        self.N = N if N is not None else self.sys.optParam['N']
        self.h = self.sys.optParam['h']
        self.Q = np.array(self.sys.optParam['Q'])
        self.R = np.array(self.sys.optParam['R'])
        self.Q_N = np.array(self.sys.optParam['Q_N'])
        self.Q_u = np.array(self.sys.optParam['Q_u'])
        self.f = self.sys.model

        self.opt_controls =  self.opti.variable(self.N, self.sys.ctrl_dim)
        self.opt_states =  self.opti.variable(self.N, self.sys.obs_dim)
        
        self.N_intp = int(self.h/self.sys.dt)
        self.traj_len = self.N * self.N_intp
        self.ctrl_full = np.zeros((self.traj_len, self.sys.ctrl_dim))
        self.states_full = np.zeros((self.traj_len, self.sys.obs_dim))

        self.opt_x0 =  self.opti.parameter(self.sys.obs_dim)
        self.opt_xs =  self.opti.parameter(self.sys.obs_dim)
        self.ctrl_goal =  self.opti.parameter(self.sys.ctrl_dim)
        self.tgt_traj =  self.opti.parameter(self.N, self.sys.obs_dim)

        self.opti.subject_to(self.getDynamicsConstraints())
        self.opti.subject_to(self.getBoundaryConstrainsts())
        self.opti.subject_to(self.getStateCtrlBounds())
        # acceptable_tol, warm_start_init_point
        opts_setting = {'ipopt.max_iter':1000, 
                        'ipopt.findiff_perturbation':1e-2,
                        'ipopt.hessian_approximation':"limited-memory",
                        'ipopt.acceptable_iter':100,  
                        'ipopt.tol': 1e-5, 
                        'ipopt.acceptable_tol': 1e-2, 
                        # 'ipopt.warm_start_init_point': 'yes',
                        'ipopt.print_level':0, #5 for more detail
                        'ipopt.print_timing_statistics':'yes', 
                        'print_time':0} 
        self.opti.solver('ipopt', opts_setting)

    def setLinearInitialValues(self, init_state, final_state):
        """
        Generate linear initial value
        @param init_state: array, inital state
        @param final_state: array, target state
        @return: array, state inital guess
        """
        state_init_guess = np.zeros((self.N, self.sys.obs_dim))
        for i in range(self.N):
            state_init_guess[i,:] = init_state + (i/( self.N - 1))* (final_state-init_state)
        return state_init_guess

    def getDynamicsConstraints(self):
        """
        Define collocation and boundary constraints
        return: constraint expression
        """
        ceq = []
        for i in range(self.N - 1): 
            ceq.append(self.getTranscriptionConstraints(self.opt_states[i,:],self.opt_states[i+1,:],
                                                    self.f(self.opt_states[i, :], self.opt_controls[i, :]),
                                                    self.h))
        # ceq.extend(self.getBoundaryConstrainsts())
        return ceq

    def getCircularConstraints(self, Cx, Cy, Cr):
        ceq = []
        for i in range(self.N): 
            for x, y, r in zip(Cx, Cy, Cr):
                ceq.append((self.opt_states[i,0] - x)**2 + (self.opt_states[i,1] - y)**2 >  (r+0.2) ** 2)
        return ceq

    def getTranscriptionConstraints(self,state1,state2,model,h):
        """
        Define collocation constraint
        @state1: array, current state
        @state2: array, next state
        @model: array, dynamics
        @h: double, step time
        return: collocation constraint expression
        """
        return (state2 == state1 + h * model)

    def getBoundaryConstrainsts(self):
        """
        Define Boundary constraints for x0 and xf
        @x0: array, initial state
        @xf: array, end state
        return: boundary state constraint expression
        """
        ceq = []
        for i in range(self.sys.obs_dim): 
            ceq.extend([(self.opt_states[0, i] == self.opt_x0[i])]) 
            if i in self.sys.sets['idx']:
                # print("terminal constraint on ", i)
                ceq.extend([(self.opt_states[-1,i] == self.opt_xs[i])]) # constraint final state to 0
          
        return ceq

    def getStateCtrlBounds(self):
        """
        Set constraints on state and control
        return: constraints expression
        """
        c = [] 
        for i in range(self.sys.obs_dim):
            c.extend([self.opti.bounded(self.sys.sets['state_min'][i], self.opt_states[:, i], self.sys.sets['state_max'][i])])

        for i in range(self.sys.ctrl_dim):
            c.extend([self.opti.bounded(self.sys.sets['u_min'][i], self.opt_controls[:, i], self.sys.sets['u_max'][i])])
        return c

    def interpolate_state_ctrl(self, x, u, dt):
        """
        Interpolate state and control 
        @x: array, state
        @u: array, control
        @dt: float, control time step
        return: (ctrl, state), u and x after interpolation
        """
        if self.N_intp == 1:
            return u, x
        
        for k in range(self.N-1):
            f_k = self.f(x[k], u[k])
            f_k1 = self.f(x[k+1], u[k+1])
            for i in range(self.N_intp):
                idx = k*self.N_intp+i
                tau = i*dt
                self.ctrl_full[idx] = u[k] + tau /self.h * (u[k+1] - u[k])
                self.states_full[idx] =  x[k] + f_k*tau + tau**2/(2*self.h)*(f_k1-f_k)

    def shift_sol(self, u_sol, x_sol, steps = 1):
        """
        Shift solution from optimizer as new initial guess
        @u_sol: array, control output from optimizer
        @x_sol: array, state output from optimizer
        @steps: int, shifting steps
        """
        # u_tile = [u_sol[-2]]*steps
        u_tile = [np.zeros(self.sys.ctrl_dim)]*steps
        x_tile = [x_sol[-1]]*steps
        u_shift = np.concatenate((u_sol[steps:], u_tile))
        x_shift = np.concatenate((x_sol[steps:], x_tile))
        return u_shift, x_shift

    def step(self, stateCurrent, stateTarget = np.zeros(4), reinit = False, shift_step = 1, ctrlTarget = [0], UGuess = [], stateGuess = [], interpolation = False, ctrl_dt=0.02):
        """
        Optimization step for fix point stablization
        """
        self.opti.set_value(self.opt_x0, stateCurrent)
        self.opti.set_value(self.opt_xs, stateTarget)
        # self.opti.set_value(self.ctrl_goal, ctrlTarget)
        if hasattr(self, 'ctrl_guess') and not reinit:
            ctrl_init_guess, state_init_guess = self.ctrl_guess, self.state_guess
        else: 
            self.ctrl_guess = np.zeros((self.N, self.sys.ctrl_dim))
            # self.state_guess = np.zeros((self.N, self.sys.obs_dim))
            self.state_guess = self.setLinearInitialValues(stateCurrent, stateTarget)
            ctrl_init_guess, state_init_guess = self.ctrl_guess, self.state_guess
        if UGuess != []:
            ctrl_init_guess = UGuess
        if stateGuess != []:
            state_init_guess = stateGuess
        # print("state_init_guess: ", state_init_guess)
        self.opti.set_initial(self.opt_controls, ctrl_init_guess)
        self.opti.set_initial(self.opt_states, state_init_guess)
        return self.solve(interpolation, dt = ctrl_dt, steps = shift_step)

    def solve(self,interpolation, dt, steps):
        """
        Solve the optimization problem
        """
        try:      
            self.sol =  self.opti.solve()
            self.ctrl_sol = self.sol.value(self.opt_controls)
            self.state_sol = self.sol.value(self.opt_states)
            self.ctrl_guess, self.state_guess = self.shift_sol(self.ctrl_sol, self.state_sol, steps = steps)
            if interpolation:  
                self.interpolate_state_ctrl(self.state_sol, self.ctrl_sol, dt = dt)
            else:
                self.ctrl_full, self.states_full = self.ctrl_sol, self.state_sol
        except:
            print("Infeasible Problem Detected!")
            return False
        return True

    def step_traj(self, stateCurrent, traj = [], interpolation = False, ctrl_dt=0.02):
        """
        Optimization step for trajectory tracking
        """
        self.opti.set_value(self.tgt_traj, traj)
        self.opti.set_initial(self.opt_controls, self.ctrl_guess)
        self.opti.set_initial(self.opt_states, self.state_guess)
        self.opti.set_value(self.opt_x0, stateCurrent)
        self.opti.set_value(self.opt_xs, traj[-1])
        return self.solve(interpolation, dt = ctrl_dt)
