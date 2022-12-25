#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@Time    :   2022/05/06 18:14:03
@Author  :   Yu Zhou 
@Contact :   yuzhou7@illinois.edu
'''

from casadi import *

class edtCostCb(Callback):
  def __init__(self, name, d, edt, opts={}):
    Callback.__init__(self)
    self.d = d
    self.edt = edt
    self.construct(name, opts)

  def get_n_in(self): return 2
  def get_n_out(self): return 1

  # Initialize the object
  def init(self):
     print('initializing object')

  def eval(self, arg):
    x, y = arg[0], arg[1]
    f = self.edt.calcDistCost([x, y], self.d)
    return [f]
