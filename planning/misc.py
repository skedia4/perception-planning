#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@Time    :   2022/05/06 18:14:59
@Author  :   Yu Zhou 
@Contact :   yuzhou7@illinois.edu
'''

import numpy as np
import time

def diff_angle(a, b):
    """returns the CCW difference between angles a and b, i.e. the amount
    that you'd neet to rotate from b to get to a.  The result is in the
    range [-pi,pi]"""
    d = a - b
    while d < -np.pi:
        d = d + np.pi * 2
    while d > np.pi:
        d = d - np.pi * 2
    return d

def identity():
    """Returns the identity transformation."""
    return ([1.,0.,0.,0.,1.,0.,0.,0.,1.],[0.,0.,0.])

def getArrowDir(s1, s2):
    arrow_vec = np.array([s1[0]-s2[0], (s1[1]-s2[1])])
    norm = np.linalg.norm(arrow_vec)
    arrow_vec = arrow_vec/norm
    return arrow_vec

def getState(x_cur, x_tgt, sys, align_heading = False):
    x_cur = np.asarray(x_cur)
    x_tgt = np.asarray(x_tgt)
    if align_heading:
        x_cur[2] = np.arctan((x_tgt[1] - x_cur[1]) / (x_tgt[0] - x_cur[0]))
    tic = time.perf_counter()
    found_solution = sys.opt.step(x_cur,x_tgt, interpolation=False, reinit = True) 
    if not found_solution:
        return [], 0, found_solution, 0
    toc = time.perf_counter()
    t = toc - tic
#     print(f"Optimization took {toc - tic:0.4f} seconds")
    print(round(sys.opt.sol.value(sys.opt.obj_tracking), 3),
          round(sys.opt.sol.value(sys.opt.obj_edt), 3),
          round(sys.opt.sol.value(sys.opt.obj_terminal), 3), 
          round(sys.opt.sol.value(sys.opt.obj_energy), 3))
    states_vis = sys.opt.sol.value(sys.opt.states_full)
    return states_vis, t, found_solution, sys.opt.sol.value(sys.opt.cost)

def pt_l_g(pt_l, scale, offset):
    pt_g = np.zeros((len(pt_l), len(pt_l[0])))
    pt_g[:, 0] = pt_l[:, 0] * scale + offset[0]
    pt_g[:, 1] = pt_l[:, 1] * scale + offset[1]
    return pt_g.astype(int)

def checkSuccess(states_vis, cc):
    pt_g = pt_l_g(states_vis)
#     print(pt_g[:, [1, 0]])
    for i in range(pt_g.shape[0]):
        if cc[pt_g[i,1]][pt_g[i,0]] < 0.8:
            print("collision detected")
            return False
    return True

def avg_t(t_arr):
    i, t_total = 0, 0
    for t in t_arr:
        if t > 10:
            continue
        t_total += t
        i += 1
    return t_total/i

def success_rate(s_arr, n):
    return np.sum(s_arr)/len(s_arr), n - len(s_arr)

def goal_dev(goals, success, states_vis_total):
    result = 0
    for i in range(len(goals)):
        if success[i]:
            result += np.linalg.norm([goals[i][0] - states_vis_total[i][-1][0], goals[i][1] - states_vis_total[i][-1][1]])
    return result/sum(success)
