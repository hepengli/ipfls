#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 20:34:11 2019

@author: lihepeng
"""
import numpy as np
from minp import Model
"""
minimize:
    f(x,y) = (x-2)**2 + 2*(y-1)**2 
subject to: 
    x+4*y <= 3
    x >= y
"""
m = Model()
x = m.addVars('x', vtype='INTEGER', n=2)
m.addConstrs('c0', ctype='INEQUALITY', c=x[0]+4*x[1]-3)
m.addConstrs('c1', ctype='INEQUALITY', c=-x[0]+x[1])
m.addObj((x[0]-2)**2+2*(x[1]-1)**2)
m.optimize()
