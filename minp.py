#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:07:56 2019

@author: lihepeng
"""

import numpy as np
from sympy import *
from ipopt import IPFLS
from scipy.sparse import csr_matrix

class Model(object):
    """ Branch and Bounding algorithm for Mixed Integer Nonlinear Programing.
    """
    def __init__(self, tol=1e-8, maxiter=100, verbosity=1):
        self.VAR_CONTINUOUS = []
        self.VAR_INTEGER = []
        self.VAR_CONTINUOUS_LB = []
        self.VAR_CONTINUOUS_UB = []
        self.VAR_INTEGER_LB = []
        self.VAR_INTEGER_UB = []

        self.CON_EQUALITY = {}
        self.CON_INEQUALITY = {}
        self.ce_lda = []
        self.ci_lda = []

        self.func = None
        self.df_func = None
        self.d2f_func = None
        self.g_func = None
        self.dg_func = None
        self.d2g_func = None
        self.h_func = None
        self.dh_func = None
        self.d2h_func = None
        self.A = None
        self.l = None
        self.u = None
        self.xmin = None
        self.xmax = None

        self.Tol = tol
        self.MaxIter = maxiter
        self.verbosity = verbosity

    def addVars(self, name, n=None, vtype='CONTINUOUS', lb=None, ub=None):

        if isinstance(name, str):
            if n is None:
                if vtype=='CONTINUOUS':
                    x = Symbol(name)
                    lb = -1e10 if lb is None else lb
                    ub = 1e10 if ub is None else ub
                    if x not in self.VAR_CONTINUOUS:
                        self.VAR_CONTINUOUS.append(x)
                        self.VAR_CONTINUOUS_LB.append(lb)
                        self.VAR_CONTINUOUS_UB.append(ub)
                elif vtype=='BINARY':
                    x = Symbol(name+'_bin')
                    if x not in self.VAR_INTEGER:
                        self.VAR_INTEGER.append(x)
                        self.VAR_INTEGER_LB.append(0.0)
                        self.VAR_INTEGER_UB.append(1.0)
                elif vtype=='INTEGER':
                    x = Symbol(name+'_int')
                    lb = -1e10 if lb is None else lb
                    ub = 1e10 if ub is None else ub
                    if x not in self.VAR_INTEGER:
                        self.VAR_INTEGER.append(x)
                        self.VAR_INTEGER_LB.append(lb)
                        self.VAR_INTEGER_UB.append(ub)
                else:
                    raise 'Please specify a variable type!'
            elif isinstance(n, int):
                if vtype=='CONTINUOUS':
                    assert (lb is None) or (len(lb)==n)
                    assert (ub is None) or (len(ub)==n)
                    lb = [-1e10]*n if lb is None else lb
                    ub = [1e10]*n if ub is None else ub
                    l = [v for v in self.VAR_CONTINUOUS if name in str(v)]
                    x = [Symbol(name+str(len(l)+i)) for i in range(n)]
                    self.VAR_CONTINUOUS.extend(x)
                    self.VAR_CONTINUOUS_LB.extend(lb)
                    self.VAR_CONTINUOUS_UB.extend(ub)
                elif vtype=='BINARY':
                    l = [n for n in self.VAR_INTEGER if name in str(n)]
                    x = [Symbol(name+str(len(l)+i)+'_bin') for i in range(n)]
                    self.VAR_INTEGER.extend(x)
                    self.VAR_INTEGER_LB.extend([0.0]*n)
                    self.VAR_INTEGER_UB.extend([1.0]*n)
                elif vtype=='INTEGER':
                    assert (lb is None) or (len(lb)==n)
                    assert (ub is None) or (len(ub)==n)
                    lb = [-1e10]*n if lb is None else lb
                    ub = [1e10]*n if ub is None else ub
                    l = [v for v in self.VAR_CONTINUOUS if name in str(v)]
                    x = [Symbol(name+str(len(l)+i)) for i in range(n)]
                    self.VAR_INTEGER.extend(x)
                    self.VAR_INTEGER_LB.extend(lb)
                    self.VAR_INTEGER_UB.extend(ub)
                else:
                    raise 'Please specify a variable type!'
            else:
                raise '"n" must be a integer!'
        elif isinstance(name, list):
            if vtype=='CONTINUOUS':
                assert (lb is None) or (len(lb)==len(name))
                assert (ub is None) or (len(ub)==len(name))
                lb = [-1e10]*len(name) if lb is None else lb
                ub = [1e10]*len(name) if ub is None else ub
                x = []
                for i, n in enumerate(name):
                    if Symbol(n) not in self.VAR_CONTINUOUS:
                        x.append(Symbol(n))
                        self.VAR_CONTINUOUS.append(Symbol(n))
                        self.VAR_CONTINUOUS_LB.append(lb[i])
                        self.VAR_CONTINUOUS_UB.append(ub[i])
            elif vtype=='BINARY':
                x = []
                for n in name:
                    if Symbol(n+'_bin') not in self.VAR_INTEGER:
                        x.append(Symbol(n+'_bin'))
                        self.VAR_INTEGER.append(Symbol(n+'_bin'))
                        self.VAR_INTEGER_LB.append(0.0)
                        self.VAR_INTEGER_UB.append(1.0)
            elif vtype=='INTEGER':
                assert (lb is None) or (len(lb)==len(name))
                assert (ub is None) or (len(ub)==len(name))
                lb = [-1e10]*len(name) if lb is None else lb
                ub = [1e10]*len(name) if ub is None else ub
                x = []
                for i, n in enumerate(name):
                    if Symbol(n) not in self.VAR_INTEGER:
                        x.append(Symbol(n))
                        self.VAR_INTEGER.append(Symbol(n))
                        self.VAR_INTEGER_LB.append(lb[i])
                        self.VAR_INTEGER_UB.append(ub[i])
            else:
                raise 'Please specify a variable type!'

        return x

    def addConstrs(self, name, ctype, c):
        if ctype=='EQUALITY':
            g = c
            x = self.VAR_CONTINUOUS+self.VAR_INTEGER
            assert simplify(g) not in self.CON_EQUALITY.values()
            assert name not in self.CON_EQUALITY.keys()
            ce_lda = Symbol('ce_lda'+str(len(self.ce_lda)))
            g = Matrix([simplify(g)])
            dg = Matrix([diff(g, v) for v in x])
            d2g = Matrix([[diff(dg_n, v) for v in x] for dg_n in dg])
            if hasattr(self, 'g'):
                self.g = self.g.row_join(g)
                self.dg = self.dg.row_join(dg)
                self.d2g = self.d2g + ce_lda * d2g
            else:
                self.g = g
                self.dg = dg
                self.d2g = ce_lda * d2g

            self.ce_lda.append(ce_lda)
            self.CON_EQUALITY[name] = g
        elif ctype=='INEQUALITY':
            h = c
            x = self.VAR_CONTINUOUS+self.VAR_INTEGER
            assert simplify(h) not in self.CON_INEQUALITY.values()
            assert name not in self.CON_INEQUALITY.keys()
            ci_lda = Symbol('ci_lda'+str(len(self.ci_lda)))
            h = Matrix([simplify(h)])
            dh = Matrix([diff(h, v) for v in x])
            d2h = Matrix([[diff(dh_n, v) for v in x] for dh_n in dh])
            if hasattr(self, 'h'):
                self.h = self.h.row_join(h)
                self.dh = self.dh.row_join(dh)
                self.d2h = self.d2h + ci_lda * d2h
            else:
                self.h = h
                self.dh = dh
                self.d2h = ci_lda * d2h

            self.ci_lda.append(ci_lda)
            self.CON_INEQUALITY[name] = h
        else:
            raise 'Please specify a correct type for the constraint!'

    def addObj(self, f):
        x = self.VAR_CONTINUOUS+self.VAR_INTEGER
        f = Matrix([simplify(f)])
        df = Matrix([diff(f, v) for v in x])
        d2f = Matrix([[diff(df_n, v) for v in x] for df_n in df])
        if hasattr(self, 'f'):
            self.f = self.f + f
            self.df = self.df + df
            self.d2f = self.d2f + d2f
        else:
            self.f = f
            self.df = df
            self.d2f = d2f

        return f

    def optimize(self):
        x = self.VAR_CONTINUOUS+self.VAR_INTEGER
        self.xmin = np.array(self.VAR_CONTINUOUS_LB + self.VAR_INTEGER_LB, dtype=np.float64)
        self.xmax = np.array(self.VAR_CONTINUOUS_UB + self.VAR_INTEGER_UB, dtype=np.float64)

        self.f_func = lambda y: np.array(self.f.subs(list(zip(x,y))), dtype=np.float64).ravel()
        self.df_func = lambda y: np.array(self.df.subs(list(zip(x,y))), dtype=np.float64).ravel()
        self.d2f_func = lambda y: csr_matrix(np.array(self.d2f.subs(list(zip(x,y))), dtype=np.float64))

        if hasattr(self, 'g'):
            self.g_func = lambda y: np.array(self.g.subs(list(zip(x,y))), dtype=np.float64).ravel()
            self.dg_func = lambda y: csr_matrix(np.array(self.dg.subs(list(zip(x,y))), dtype=np.float64))
            self.d2g_func = lambda y, z: csr_matrix(np.array(
                self.d2g.subs(list(zip(x,y))+list(zip(self.ce_lda,z))), dtype=np.float64))

        if hasattr(self, 'h'):
            self.h_func = lambda y: np.array(self.h.subs(list(zip(x,y))), dtype=np.float64).ravel()
            self.dh_func = lambda y: csr_matrix(np.array(self.dh.subs(list(zip(x,y))), dtype=np.float64))
            self.d2h_func = lambda y, z: csr_matrix(np.array(
                self.d2h.subs(list(zip(x,y))+list(zip(self.ci_lda,z))), dtype=np.float64))

        self.x0 = self.xmin+np.random.uniform(0.49,0.51)*(self.xmax-self.xmin)
        p = IPFLS(x0=self.x0, xmin=self.xmin, xmax=self.xmax,
                  f_func=self.f_func, df_func=self.df_func, d2f_func=self.d2f_func,
                  g_func=self.g_func, dg_func=self.dg_func, d2g_func=self.d2g_func,
                  h_func=self.h_func, dh_func=self.dh_func, d2h_func=self.d2h_func,
                  float_dtype=np.float64, verbosity=self.verbosity)
        opt_x, s, lda, z, fval, signal = p.solve()

        if self.VAR_INTEGER.__len__() != 0:
            itertion = 0
            while True:
                x_int = opt_x[-self.VAR_INTEGER.__len__():]
                if all((x_int-np.round(x_int))<self.Tol):
                    break
                else:
                    vint_ind = [i for i, xi in enumerate(x_int) if xi-np.round(xi)>self.Tol][0]
                ind = self.VAR_CONTINUOUS.__len__() + vint_ind

                # Branch 1
                p.xmin = np.array(self.VAR_CONTINUOUS_LB + self.VAR_INTEGER_LB, dtype=np.float64)
                p.xmax = np.array(self.VAR_CONTINUOUS_UB + self.VAR_INTEGER_UB, dtype=np.float64)
                p.xmin[ind] = np.ceil(opt_x[ind])
                p.x0 = p.xmin+np.random.uniform(0.49,0.51)*(p.xmax-p.xmin)
                p.compile()
                opt_x1, s1, lda1, z1, fval1, signal1 = p.solve()
                # Branch 2
                p.xmin = np.array(self.VAR_CONTINUOUS_LB + self.VAR_INTEGER_LB, dtype=np.float64)
                p.xmax = np.array(self.VAR_CONTINUOUS_UB + self.VAR_INTEGER_UB, dtype=np.float64)
                p.xmax[ind] = np.floor(opt_x[ind])
                p.x0 = p.xmin+np.random.uniform(0.49,0.51)*(p.xmax-p.xmin)
                p.compile()
                opt_x2, s2, lda2, z2, fval2, signal2 = p.solve()
                # Bound
                if signal1==1 and signal2==1:
                    if fval1 < fval2:
                        self.VAR_INTEGER_LB[vint_ind] = np.ceil(opt_x[ind]).copy()
                    else:
                        self.VAR_INTEGER_UB[vint_ind] = np.floor(opt_x[ind]).copy()
                elif signal1==1:
                    self.VAR_INTEGER_LB[vint_ind] = np.ceil(opt_x[ind]).copy()
                elif signal2==1:
                    self.VAR_INTEGER_UB[vint_ind] = np.floor(opt_x[ind]).copy()
                else:
                    signal = -1
                    break
                # resolve the problem with new bounds
                p.xmin = np.array(self.VAR_CONTINUOUS_LB + self.VAR_INTEGER_LB, dtype=np.float64)
                p.xmax = np.array(self.VAR_CONTINUOUS_UB + self.VAR_INTEGER_UB, dtype=np.float64)
                p.x0 = p.xmin+np.random.uniform(0.49,0.51)*(p.xmax-p.xmin)
                p.compile()
                opt_x, s, lda, z, fval, signal = p.solve()

                itertion += 1
                if itertion > self.MaxIter:
                    break

        self.x = opt_x
        self.fval = fval
        self.signal = signal

        if self.verbosity >= 1:
            if signal == 1:
                msg = "Congratulations! Converged to Ktol tolerance! "
            elif signal == -1:
                msg = "Maximum iterations reached!"
            else:
                msg = "Problem Infeasible!"
            print(msg)
            print(self.x, self.fval)
