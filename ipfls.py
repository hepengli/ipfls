# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:42:42 2017

@author: lihepeng
"""

import numpy as np
from numpy.linalg import norm, pinv, cond, cholesky
from numpy import minimum, maximum, flatnonzero as find, concatenate as stack

from scipy.sparse import vstack, hstack, eye, diags, csr_matrix
from scipy.sparse.linalg import spsolve 

class IPFLS:
    """Interior point filter line search for NLP (nonlinear programming).
    Minimize a function F(X) beginning from a starting point M{x0}, subject to
    optional linear and nonlinear constraints and variable bounds::

            min f(x)
             x

    subject to:

            g(x) = 0            (nonlinear equalities)
            h(x) <= 0           (nonlinear inequalities)
            l <= A*x <= u       (linear constraints)
            xmin <= x <= xmax   (variable bounds)
    """
    def __init__(self, x0=None, A=None, l=None, u=None, xmin=None, xmax=None,
                 f_func=None, df_func=None, d2f_func=None, 
                 g_func=None, dg_func=None, d2g_func=None, 
                 h_func=None, dh_func=None, d2h_func=None, 
                 s0=None, lda0=None, z0=None, 
                 mu0=0.1, miter=20, niter=10, 
                 Etol=1.0E-8, Ktol=10.0,
                 float_dtype=np.float64, verbosity=1):
        
        self.x0 = x0
        self.s0 = s0
        self.lda0 = lda0
        self.z0 = z0
        
        self.A = A
        self.l = l
        self.u = u
        self.xmin = xmin
        self.xmax = xmax
        
        self.f_func = f_func
        self.df_func = df_func
        self.d2f_func = d2f_func
        self.g_func = g_func
        self.dg_func = dg_func
        self.d2g_func = d2g_func
        self.h_func = h_func
        self.dh_func = dh_func
        self.d2h_func = d2h_func
        
        self.k1 = 1e-2
        self.k2 = 1e-2
        
        self.smax = 100.0
        
        self.eps = np.finfo(float_dtype).eps
        self.Ktol = Ktol
        self.Etol = Etol
        self.Xtol = self.eps
        
        self.k_mu = 0.2
        self.theta_mu = 1.5
        self.tau_min = 0.99
        self.k_sigma = 1e10
        self.gamma_theta = 1e-5
        self.gamma_phi = 1e-5
        self.delta = 1.0
        self.gamma_alpha = 0.05
        self.s_theta = 1.1
        self.s_phi = 2.3
        self.eta_phi = 1e-4
        self.k_soc = 0.99
        self.p_max = 4
        
        self.delta_wmin = 1e-20
        self.delta_w0_bar = 1e-4
        self.delta_wmax = 1e40
        self.delta_c_bar = float_dtype(np.sqrt(self.eps))
        self.k_wmin = 0.3333
        self.k_wmax = 8.0
        self.k_wmax_bar = 100.0
        self.k_c = 0.25
        self.delta_w_last = 0.0
        self.delta_w0 = 0.0
        
        self.mu0 = mu0
        self.mu = np.copy(self.mu0)
        self.tau = np.max([self.tau_min, 1-self.mu])
        self.miter = miter
        self.niter = niter
        
        self.verbosity = verbosity
        self.float = float_dtype

        self.numpy_printoptions = np.set_printoptions(precision=4)

        self.compiled = False
        
    def validate(self):
        """Validate inputs

        """
        
        assert self.f_func is not None
        assert (self.g_func is not None) or \
               (self.g_func is None and self.dg_func is None and self.d2g_func is None)
        assert (self.h_func is not None) or \
               (self.h_func is None and self.dh_func is None and self.d2h_func is None)
        assert (self.A is None and self.l is None and self.u is None) or \
               (self.A is not None and (self.l is not None or self.u is not None))
        
        assert self.k1 > 0.0
        assert self.k2 > 0.0 and self.k2 < 0.5 
        assert self.smax >= 1.0
        
        assert self.Ktol > 0.0
        assert self.Etol >= self.eps
        assert self.Xtol >= self.eps
        
        assert self.k_mu > 0.0 and self.k_mu < 1.0
        assert self.theta_mu > 1.0 and self.theta_mu < 2.0
        assert self.tau_min > 0.0 and self.tau_min > 0.0
        assert self.k_sigma > 1
        assert self.gamma_theta > 0.0 and self.gamma_theta < 1.0
        assert self.gamma_phi > 0.0 and self.gamma_phi < 1.0
        assert self.delta > 0.0
        assert self.gamma_alpha > 0.0 and self.gamma_alpha <= 1
        assert self.s_theta > 1.
        assert self.s_phi >= 1.0
        assert self.eta_phi > 0.0 and self.eta_phi < 0.5
        assert self.k_soc > 0.0 and self.k_soc < 1 
        
        assert self.delta_wmin > 0.0
        assert self.delta_w0_bar > self.delta_wmin
        assert self.delta_wmax > self.delta_w0_bar
        assert self.delta_c_bar > 0.0
        assert self.k_wmin > 0.0 and self.k_wmin < 1.0
        assert self.k_wmax > 1.0
        assert self.k_wmax_bar > self.k_wmax
        assert self.k_c >= 0.0
        
        assert self.mu > 0.0
        assert self.tau > 0.0 and self.tau < 1.0
        assert self.miter >= 0 and isinstance(self.miter, int)
        assert self.niter >= 0 and isinstance(self.miter, int)

    def compile(self):
        """
        Validate some of the input variables and compile the objective 
        function, the gradient, and the Hessian with constraints.
        
        """

        # set the variable counter equal to the number of variables
        nvar = self.x0.size

        # set linear constraints
        nA = self.A.shape[0] if self.A is not None else 0
        if self.l is None: 
            l = -np.Inf * np.ones(nA)
        else:
            l = self.l
        if self.u is None: 
            u = np.Inf * np.ones(nA)
        else:
            u = self.u
        if self.xmin is None: 
            xmin = -np.Inf * np.ones(self.x0.shape[0])
        else:
            xmin = self.xmin
        if self.xmax is None: 
            xmax =  np.Inf * np.ones(self.x0.shape[0])
        else:
            xmax = self.xmax

        eyex = eye(nvar, nvar, format='csr', dtype=self.float)
        AA = eyex if self.A is None else vstack([eyex, self.A], "csr")
        ll = stack([xmin, l])
        uu = stack([xmax, u])

        ieq = find( np.abs(uu - ll) <= self.eps )
        igt = find( (uu >=  1e10) & (ll > -1e10) )
        ilt = find( (ll <= -1e10) & (uu <  1e10) )
        ibx = find( (np.abs(uu - ll) > self.eps) & (uu < 1e10) & (ll > -1e10) )

        Ae = AA[ieq, :] if len(ieq) else None
        if len(ilt) or len(igt) or len(ibx):
            idxs = [(1, ilt), (-1, igt), (1, ibx), (-1, ibx)]
            Ai = vstack([sig * AA[idx, :] for sig, idx in idxs if len(idx)], 'csr')
        else:
            Ai = None
        be = uu[ieq]
        bi = stack([uu[ilt], -ll[igt], uu[ibx], -ll[ibx]])

        f = self.f_func
        df = self.df_func
        d2f = self.d2f_func

        # get the number of equality constraints
        if self.g_func is not None and Ae is not None:
            ce = lambda x: stack([self.g_func(x), Ae * x - be])
            dce = lambda x: hstack([self.dg_func(x), Ae.T])
            d2ce = self.d2g_func
            neq = ce(self.x0).size
        elif self.g_func is not None:
            ce = self.g_func
            dce = self.dg_func
            d2ce = self.d2g_func
            neq = ce(self.x0).size
        elif Ae is not None:
            ce = lambda x: Ae * x - be
            dce = lambda x: csr_matrix(Ae.T)
            d2ce = lambda x, lda: 0
            neq = ce(self.x0).size
        else:
            neq = 0
        
        # get the number of inequality constraints
        if self.h_func is not None and Ai is not None:
            ci = lambda x: stack([self.h_func(x), Ai * x - bi])
            dci = lambda x: hstack([self.dh_func(x), Ai.T])
            d2ci = self.d2h_func
            niq = ci(self.x0).size
        elif self.h_func is not None:
            ci = self.h_func
            dci = self.dh_func
            d2ci = self.d2h_func
            niq = ci(self.x0).size
        elif Ai is not None:
            ci = lambda x: Ai * x - bi
            dci = lambda x: csr_matrix(Ai.T)
            d2ci = lambda x, lda: 0
            niq = ci(self.x0).size
        else:
            niq = 0

        # construct composite expression for the constraints
        if neq and niq:
            con = lambda x, s: stack([ce(x), ci(x) + s])
        elif neq:
            con = lambda x, s: ce(x)
        elif niq:
            con = lambda x, s: ci(x) + s
        else:
            con = None
        
        # construct composite expression for the constraints Jacobian
        if neq and niq:
            jaco_top = lambda x: hstack([dce(x), dci(x)])
            jaco_bot = hstack([csr_matrix((niq,neq)), eye(niq)])
            jaco = lambda x: vstack([jaco_top(x), jaco_bot], format='csr')
        elif neq:
            jaco = lambda x: csr_matrix(dce(x))
        elif niq:
            jaco = lambda x: vstack([dci(x), eye(niq)], format='csr')
        else:
            jaco = None
        
        # construct expressions for the merit function
        if niq:
            phi = lambda x, s: f(x) - self.mu * np.sum(np.log(s+self.eps))
        else:
            phi = lambda x, s: f(x)
        
        # construct expressions for the merit function gradient
        if niq:
            dphi = lambda x, s: stack([df(x), -self.mu/(s+self.eps)])
        else:
            dphi = lambda x, s: df(x)
        
        # construct expression for the gradient
        if neq or niq:
            grad = lambda x, s, lda: stack([dphi(x, s) + jaco(x).dot(lda),
                                            con(x, s)], axis=0)
        else:
            grad = lambda x, s, lda: dphi(x, s)

        # construct expression for the Hessian of the Lagrangian
        if neq and niq:
            d2L = lambda x, lda: d2f(x) + d2ce(x, lda[:neq]) + d2ci(x, lda[neq:])
        elif neq:
            d2L = lambda x, lda: d2f(x) + d2ce(x, lda)
        elif niq:
            d2L = lambda x, lda: d2f(x) + d2ci(x, lda)
        else:
            d2L = lambda x, lda: d2f(x) 
            
        if niq:
            Sigma = lambda s, z: diags(z/(s+self.eps), format='csr')
            hess_11 = lambda x, s, lda, z: \
                      vstack([hstack([d2L(x,lda), csr_matrix((nvar, niq))]),
                              hstack([csr_matrix((niq, nvar)), Sigma(s,z)])], 
                              format='csr')
        else:
            hess_11 = lambda x, s, lda, z: d2L(x,lda)
        
        if niq or neq:
            hess_12 = lambda x: jaco(x)
            hess_21 = lambda x: jaco(x).transpose()
            hess_22 = csr_matrix((neq+niq, neq+niq))
            hess = lambda x, s, lda, z: \
                   vstack([hstack([hess_11(x,s,lda,z), hess_12(x)]),
                                   hstack([hess_21(x), hess_22])], format='csr')
        else:
            hess = lambda x, s, lda, z: hess_11(x, s, lda, z)
            
        # construct expression for initializing the decision variables
        if (self.xmin is not None) and (self.xmax is not None):
            xmax[find(xmax >=  1e10)] = 1e10
            xmin[find(xmin <= -1e10)] = -1e10
            PL1 = self.k1 * maximum(1.0, np.abs(xmin))
            PL2 = self.k2 * (xmax - xmin)
            PU1 = self.k1 * maximum(1.0, np.abs(xmax))
            PU2 = self.k2 * (xmax - xmin)
            PL = minimum(PL1, PL2)
            PU = minimum(PU1, PU2)
            init_x = lambda x: minimum( maximum(x, xmin+PL), xmax-PU )
        elif self.xmin is not None:
            xmin[find(xmin <= -1e10)] = -1e10
            PL = self.k1 * maximum(1.0, np.abs(xmin))
            init_x = lambda x: maximum(x, self.xmin + PL)
        elif self.xmax is not None:
            xmax[find(xmax >=  1e10)] = 1e10
            PU = self.k1 * maximum(1.0, np.abs(xmax))
            init_x = lambda x: minimum(x, xmax - PL)
        else:
            init_x = lambda x: x
        
        # construct expression for initializing the slack variables
        if niq:
            init_slack = lambda x: maximum(-ci(x), self.Etol)
            
            z1 = lambda s: self.k_sigma*self.mu/(s+self.eps)
            z2 = lambda s: self.mu/(self.k_sigma*s+self.eps)
            correct_z = lambda z, s: maximum(minimum(z, z1(s)), z2(s))
        
        # construct expression for initializing the Lagrange multipliers
        if neq or niq:
            pinv_jaco = lambda x: pinv(jaco(x).toarray()[:nvar,:])
            init_lda = lambda x: np.dot(pinv_jaco(x), df(x))
        
        self.neq = neq
        self.niq = niq

        # compile expressions into device functions
        self.cost = self.f_func
        self.grad = grad
        self.hess = hess
        self.phi = phi
        self.dphi = dphi
        self.init_x = init_x
        if neq or niq:
            self.con = con
            self.jaco = jaco
            self.init_lda = init_lda
        if niq:
            self.init_slack = init_slack
            self.correct_z = correct_z

        self.compiled = True

    def KKT(self, x, s, lda, z):
        """Calculate the first-order Karush-Kuhn-Tucker conditions. 
           Irrelevant conditions are set to zero.

        """
        if self.neq or self.niq:
            if self.niq:
                m = lda.size
                n = s.size
                sd = np.max([self.smax, (norm(lda,1)+norm(z,1))/(m+n)]) / self.smax
                sc = np.max([self.smax, norm(z,1)/n]) / self.smax

                gradient = norm(stack([self.df_func(x), -z]) + self.jaco(x).dot(lda), np.inf)
                constraint = norm(self.con(x, s), np.inf)
                barrier = norm(s * z - self.mu, np.inf)

                E_mu = np.max([gradient/sd, constraint, barrier/sc])
            else:
                m = lda.size
                sd = np.max([self.smax, norm(lda,1)/m]) / self.smax

                gradient = norm(self.df_func(x) + self.jaco(x).dot(lda), np.inf)
                constraint = norm(self.con(x, s), np.inf)
                barrier = self.float(0.0)

                E_mu = np.max([gradient/sd, constraint, barrier])
        else:
            gradient = norm(self.df_func(x), np.inf)
            constraint = self.float(0.0)
            barrier = self.float(0.0)

            E_mu = np.max([gradient, constraint, barrier])

        return E_mu


    def reghess(self, H):
        """Regularize the Hessian to avoid ill-conditioning and to escape saddle
           points.

        """
        H = H.todense()
        # check the inertia and condition number of the Hessian matrix
        rcond = cond(H);
        H11 = H[:self.nvar+self.niq, :self.nvar+self.niq]
        try:
            cholesky(H11)
            p = 0
        except:
            p = -1

        if rcond >= 1/self.eps or p < 0:
            # if the Hessian is ill-conditioned or the matrix inertia is undesireable, regularize the Hessian
            if rcond <= self.eps and self.neq:
                self.delta_c = self.delta_c_bar * (self.mu**self.k_c)
                ind1 = self.nvar+self.niq
                ind2 = ind1+self.neq
                H[ind1:ind2, ind1:ind2] -= self.delta_c * np.eye(self.neq)
                
            if self.delta_w_last == 0.0:
                self.delta_w = self.delta_w0_bar
            else:
                self.delta_w = np.max([self.delta_wmin, self.k_wmin * self.delta_w_last])

            # regularize Hessian with diagonal shift matrix (delta*I)
            H[:self.nvar, :self.nvar] += self.delta_w * np.eye(self.nvar)
            H11 = H[:self.nvar+self.niq, :self.nvar+self.niq]
            try:
                cholesky(H11)
                p = 0
            except:
                p = -1

            while p < 0:
                H[:self.nvar, :self.nvar] -= self.delta_w * np.eye(self.nvar)
                if self.delta_w_last == 0.0:
                    self.delta_w *= self.k_wmax_bar
                else:
                    self.delta_w *= self.k_wmax
                # preventing the delta_w being too big
                if self.delta_w > self.delta_wmax:
                    print('GO TO STEP A-9')
                    break
                H[:self.nvar, :self.nvar] += self.delta_w * np.eye(self.nvar)
                H11 = H[:self.nvar+self.niq, :self.nvar+self.niq]
                try:
                    cholesky(H11)
                    p = 0
                except:
                    p = -1

            self.delta_w_last = self.delta_w
        # return regularized Hessian
        H = csr_matrix(H)
        return H

    def step(self, x, dx):
        """Golden section search used to determine the maximum
           step length for slack variables and Lagrange multipliers
           using the fraction-to-the-boundary rule.

        """

        GOLD = (np.sqrt(5)+1.0)/2.0

        a = 0.0
        b = 1.0
        if np.all(x + b*dx >= (1.0 - self.tau) * x):
            return b
        else:
            c = b - (b - a)/GOLD
            d = a + (b - a)/GOLD
            while np.abs(b - a) > GOLD * self.Xtol:
                if np.any(x + d * dx < (1.0 - self.tau) * x):
                    b = np.copy(d)
                else:
                    a = np.copy(d)
                if c > a:
                    if np.any(x + c * dx < (1.0 - self.tau) * x):
                        b = np.copy(c)
                    else:
                        a = np.copy(c)

                c = b - (b - a)/GOLD
                d = a + (b - a)/GOLD

            return self.float(a)
    
    def search(self, x0, s0, dk, H, B, alpha_smax):
        """Backtracking line search to find a solution that leads
           to a smaller value of the Lagrangian within the confines
           of the maximum step length for the slack variables and
           Lagrange multipliers found using class function 'step'.

        """

        # get some dimensions
        nx = self.nvar
        ns = self.niq

        # extract search directions along x, s
        dx = dk[:nx]
        ds = dk[nx:(nx+ns)]
        dxs = dk[:(nx+ns)]

        # compute the filter condition wrt x0, s0
        theta_0 = norm(self.con(x0, s0), 1)
        phi_0 = self.phi(x0, s0)
        dphi_0 = np.dot(self.dphi(x0, s0).T, dxs)

        # calculate the minimum step size
        if (dphi_0 < -self.eps) and (theta_0 <= self.theta_min):
            AM1 = self.gamma_theta
            AM2 = self.gamma_phi * theta_0 / (-dphi_0)
            AM3 = self.delta*(theta_0**self.s_theta) / ((-dphi_0)**self.s_phi)                
            alpha_min = self.gamma_alpha * np.min([AM1, AM2, AM3])
        elif (dphi_0 < -self.eps) and (theta_0 > self.theta_min):
            AM1 = self.gamma_theta
            AM2 = self.gamma_phi* theta_0 / (-dphi_0)
            alpha_min = self.gamma_alpha * np.min([AM1, AM2])
        else:
            alpha_min = self.gamma_alpha

        # initialize the line search
        l = 0 
        alpha = np.copy(alpha_smax)
        accept = False

        # starts backtracking line search
        while not accept:
            # compute the new trial point and the filter condition
            xl = x0 + alpha*dx
            sl = s0 + alpha*ds
            theta_l = norm(self.con(xl, sl), 1)
            phi_l = self.phi(xl, sl)

            # check acceptability to the filter
            reject = (theta_l >= self.theta_max) and (phi_l >= self.phi_max)

            # check sufficiet derease wrt the current iterate
            if not reject:
                if (phi_l <= phi_0 + self.eta_phi * alpha * dphi_0):
                    accept = True
                    return alpha, accept
                if (theta_l <= (1-self.gamma_theta) * theta_0) or \
                   (phi_l <= phi_0 - self.gamma_phi * theta_0):
                    accept = True
                    return alpha, accept

            # Initialize the second-order correction
            skip_soc = (l > 0) or (theta_l < theta_0)

            if not skip_soc:
                # initialize SOC counter p
                p = 1
                ck_soc = alpha * self.con(x0,s0) + self.con(xl,sl)
                theta_old_soc = theta_0

                while p <= self.p_max:
                    # compute the second-order correction
                    B[(nx+ns):] = -ck_soc
                    dk_soc = spsolve(H, B).reshape((B.size,))
                    dx_soc = dk_soc[:nx]
                    ds_soc = dk_soc[nx:(nx+ns)]
                    if self.niq:
                        alpha_soc = self.step(s0, ds_soc)
                    else:
                        alpha_soc = self.float(1.0)

                    # compute the soc trial point and the filter condition
                    x_soc = x0 + alpha_soc*dx_soc
                    s_soc = s0 + alpha_soc*ds_soc
                    theta_soc = norm(self.con(x_soc, s_soc), 1)
                    phi_soc = self.phi(x_soc, s_soc)

                    # check acceptability to the filter in soc
                    reject_soc = (theta_soc >= self.theta_max) and \
                                 (phi_soc >= self.phi_max)
                    if reject_soc:
                        break

                    # check sufficiet derease wrt the current iterate in soc
                    if (phi_soc <= phi_0 + self.eta_phi * alpha * dphi_0):
                        accept = True
                        return alpha_soc, accept
                    if (theta_soc <= (1-self.gamma_theta) * theta_0) or \
                       (phi_soc <= phi_0 - self.gamma_phi * theta_0):
                        accept = True
                        return alpha_soc, accept                     

                    # next second-order correction
                    if theta_soc > self.k_soc * theta_old_soc:
                        break
                    else:
                        p += 1
                        ck_soc = alpha_soc * ck_soc + self.con(x_soc, s_soc)
                        theta_old_soc = theta_soc

            # choose the new trial step size
            alpha *= 0.5
            l += 1
            if alpha <= alpha_min:
                accept = False
                return alpha, accept

        return alpha, accept

    def solve(self, x0=None, s0=None, lda0=None, z0=None, force_recompile=False):
        """Main solver function that initiates, controls the iteraions, and
         performs the primary operations of the line search primal-dual
         interior-point method.         

        """

        # check if variables, slacks, or multipliers are given
        if x0 is not None:
            self.x0 = x0
        if s0 is not None:
            self.s0 = s0
        if lda0 is not None:
            self.lda0 = lda0
        if z0 is not None:
            self.z0 = z0

        # variables must be initialized and have length greater than zero
        assert (self.x0 is not None) and (self.x0.size > 0)
        # variables should be a one-dimensional array
        assert self.x0.size == self.x0.shape[0]
        # set the variable counter equal to the number of variables
        self.nvar = self.x0.size
        # cast variables to float_dtype
        self.x0 = self.float(self.x0)

        # validate class members
        self.validate()

        # compile expressions into functions
        if not self.compiled:
            self.compile()

        # if complied, re-initialize the barrier and the "fraction-to-the-boundary" parameter
        if self.compiled:
            self.mu = np.copy(self.mu0)
            self.tau = np.max([self.tau_min, 1-self.mu])

        # initialize variables, slacks, and multipliers
        x = self.init_x(self.x0)

        if self.niq:
            if self.s0 is None:
                s = self.init_slack(x)
                z = np.copy(s)
            else:
                s = self.s0.astype(self.float)
                z = self.z0.astype(self.float)
        else:
            s = np.array([], dtype=self.float)
            z = np.array([], dtype=self.float)
            
        if self.neq or self.niq:
            if self.lda0 is None:
                lda = self.init_lda(x)
                if self.niq and self.neq:
                    lda_ieq = lda[self.neq:]
                    lda_ieq[lda_ieq < self.float(0.0)] = self.float(self.Etol)
                    lda[self.neq:] = lda_ieq
                elif self.niq:
                    lda[lda < self.float(0.0)] = self.float(self.Etol)
            else:
                lda = self.lda0.astype(self.float)
        else:
            lda = np.array([], dtype=self.float)

        # initialize the filter
        if self.neq or self.niq:
            self.theta_max0 = 1e4 * np.max([1, norm(self.con(x, s), 1)])
            self.theta_min0 = 1e-4 * np.max([1, norm(self.con(x, s), 1)])
            self.phi_max0 = self.phi(x, s)

            self.theta_max = np.copy(self.theta_max0)
            self.theta_min = np.copy(self.theta_min0)
            self.phi_max = np.copy(self.phi_max0)

        # inilialize the delta_w
        self.delta_w = np.copy(self.delta_w0)
        self.delta_w_last = np.copy(self.delta_w0)

        # calculate the initial KKT conditions
        E_mu = self.KKT(x, s, lda, z)

        # initialize optimization return signal
        self.signal = 0

        # starts to loop
        iter_count = 0

        for outer in range(self.niter):
            # Check convergence for the overall problem
            if E_mu <= self.Etol:
                self.signal = 1
                break

            if self.verbosity > 1 and self.niq:
                print("OUTER ITERATION " + str(outer+1))

            for inner in range(self.miter):
                # check convergence for the barrier problem
                if E_mu <= self.Ktol * self.mu:
                    break

                if self.verbosity > 1:
                    if self.niq:
                        msg = "* INNER ITERATION " + str(inner+1)
                    else:
                        msg = "ITERATION " + str(iter_count+1)
                    if self.verbosity > 2:
                        msg += ", f(x) = " + str(self.cost(x))
                    print(msg)

                # compute the search direction
                H = self.reghess(self.hess(x, s, lda, z))
                B = - self.grad(x, s, lda)
                dk = spsolve(H, B).reshape((B.size,))
                dx = dk[:self.nvar]
                dl = dk[(self.nvar+self.niq):]
                if self.niq:
                    ds = dk[self.nvar:(self.nvar+self.niq)]
                    dz = self.mu/(s+self.eps) - z - ds * z/(s+self.eps)

                # use fraction-to-the-boundary rule to compute the step size
                if self.niq:
                    alpha_smax = self.step(s, ds)
                    alpha_z = self.step(z, dz)
                else:
                    alpha_smax = self.float(1.0)

                # backtracking line search
                if self.neq or self.niq:
                    alpha_s, accept = self.search(x, s, dk, H, B, alpha_smax)
                else:
                    alpha_s = self.float(1.0)
                    accept = True

                #  compute the filter wrt the current x, s
                if self.neq or self.niq:
                    theta_old = norm(self.con(x, s), 1)
                    phi_old = self.phi(x, s)
                    dphi_old = np.dot(self.dphi(x, s), dk[:(self.nvar+self.niq)])

                # Accept the trial point and update x, s, lda, z
                if accept:
                    x = x + alpha_s*dx
                    lda = lda + alpha_s*dl
                    if self.niq:
                        s = s + alpha_s*ds
                        z = z + alpha_z*dz
                        z = self.correct_z(z, s)

                # augment the filter
                if self.neq or self.niq:
                    phi_now = self.phi(x, s)
                    if phi_now > phi_old+self.eta_phi*alpha_s*dphi_old:
                        self.theta_max = min(self.theta_max, (1-self.gamma_theta)*theta_old)
                        self.phi_max = min(self.phi_max, phi_old-self.gamma_phi*theta_old)

                # update iteration counter
                iter_count += 1

                # check the KKT condition
                E_mu = self.KKT(x, s, lda, z)

            # update the barrier parameter
            mu = np.min([self.k_mu*self.mu, self.mu**self.theta_mu])
            self.mu = np.max([self.Etol/10.0, mu])

            if self.neq or self.niq:
                # update the "fraction-to-the-boundary" parameter
                self.tau = np.max([self.tau_min, 1-self.mu])
                # re-initialize the filter
                self.theta_max = np.copy(self.theta_max0)
                self.phi_max = np.copy(self.phi_max0)

            if outer >= self.niter-1:
                self.signal = -1
                if self.verbosity > 1:
                    if self.niq:
                        print("MAXIMUM OUTER ITERATIONS EXCEEDED")
                    else:
                        print("MAXIMUM ITERATIONS EXCEEDED")
                break

        # assign class member variables to the solutions
        self.x = x
        self.s = s
        self.lda = lda
        self.z = z
        self.fval = self.cost(x)

        if self.verbosity > 0:
            if self.signal == 1:
                msg = "Congratulations! Converged to Ktol tolerance! "
            else:
                msg = "Maximum iterations reached "
                outer = self.niter
                inner = 0
            print(msg)

        # return solution weights, slacks, multipliers, KKT conditions, and cost
        return (self.x, self.s, self.lda, self.z, self.fval, self.signal)

def main():
    prob = 11
    float_dtype = np.float64
    if prob == 1:
        print("minimize f(x,y) = x**2 - 4*x + y**2 - y - x*y")
        def f_func(x):
            f = x[0]**2 - 4*x[0] + x[1]**2 - x[1] - x[0]*x[1]
            return f
        def df_func(x):
            df = np.array([2.0*x[0]-4.0-x[1], 2.0*x[1]-1.0-x[0]])
            return df
        def d2f_func(x):
            d2f = np.array([[2.0,-1.0],
                            [-1.0,2.0]])
            d2f = csr_matrix(d2f)
            return d2f
        x0 =  np.random.randn(2).astype(float_dtype)
        p = IPFLS(x0=x0, f_func=f_func, df_func=df_func, d2f_func=d2f_func,
                  float_dtype=float_dtype)
        x, s, lda, z, fval, signal = p.solve()
        print(x)
    elif prob == 2:
        print("minimize f(x,y) = -(x**2)*y subject to x**2 + y**2 = 3")
        def f_func(x):
            f = -(x[0]**2)*x[1]
            return f
        def df_func(x):
            df = np.array([-2.0*x[0]*x[1], -x[0]**2])
            return df
        def d2f_func(x):
            d2f = np.array([[-2.0*x[1],-2.0*x[0]],
                            [-2.0*x[0],0.0]])
            d2f = csr_matrix(d2f)
            return d2f
        def g_func(x):
            g = np.array([x[0]**2 + x[1]**2 - 3.0])
            return g
        def dg_func(x):
            dg = np.array([[2.0*x[0]],
                           [2.0*x[1]]])
            return dg
        def d2g_func(x, lda):
            d2g = lda[0] * np.array([[2.0, 0.0],
                                     [0.0, 2.0]])
            d2g = csr_matrix(d2g)
            return(d2g)
        x0 =  np.random.randn(2).astype(float_dtype)
        p = IPFLS(x0=x0,
                  f_func=f_func, df_func=df_func, d2f_func=d2f_func,
                  g_func=g_func, dg_func=dg_func, d2g_func=d2g_func,
                  float_dtype=float_dtype)
        x, s, lda, z, fval, signal = p.solve()
        print(x)
    elif prob == 3:
        print("minimize f(x,y) = x**2 + 2*y**2 + 2*x + 8*y subject to -x-2*y+10 <= 0, x >= 0, y >= 0")
        def f_func(x):
            f = np.array([x[0]**2 + 2.0*x[1]**2 + 2.0*x[0] + 8.0*x[1]])
            return f
        def df_func(x):
            df = np.array([2.0*x[0] + 2.0, 4.0*x[1] + 8.0])
            return df
        def d2f_func(x):
            d2f = np.array([[2.0, 0.0],
                            [0.0, 4.0]])
            d2f = csr_matrix(d2f)
            return d2f
        def h_func(x):
            h = np.array([-x[0]-2.0*x[1]+10.0])
            return h
        def dh_func(x):
            dh = np.array([[-1.0],
                           [-2.0]])
            dh = csr_matrix(dh)
            return dh
        def d2h_func(x, lda):
            d2h = lda[0] * np.array([[0.0, 0.0],
                                     [0.0, 0.0]])
            d2h = csr_matrix(d2h)
            return(d2h)
        x0 =  np.random.randn(2).astype(float_dtype)
        xmin = np.array([0,0]).astype(float_dtype)
        p = IPFLS(x0=x0, xmin=xmin, A=np.array([[-1.0,-2.0]]), u=np.array([-10.0]),#l=np.array([-np.inf]),
                  f_func=f_func, df_func=df_func, d2f_func=d2f_func,
#                  h_func=h_func, dh_func=dh_func, d2h_func=d2h_func,
                  float_dtype=float_dtype)
        x, s, lda, z, fval, signal = p.solve()
        print(x)
    elif prob == 4:
        print("minimize f(x,y) = (x-2)**2 + 2*(y-1)**2 subject to x+4*y <= 3, x >= y")
        def f_func(x):
            f = np.array([(x[0]-2.0)**2 + 2.0*(x[1]-1.0)**2])
            return f
        def df_func(x):
            df = np.array([2.0*(x[0]-2.0), 4.0*(x[1]-1.0)])
            return df
        def d2f_func(x):
            d2f = np.array([[2.0, 0.0],
                            [0.0, 4.0]])
            d2f = csr_matrix(d2f)
            return d2f
        def h_func(x):
            h = np.array([x[0]+4.0*x[1]-3.0, -x[0]+x[1]])
            return h
        def dh_func(x):
            dh = np.array([[1.0, -1.0],
                           [4.0, 1.0]])
            dh = csr_matrix(dh)
            return dh
        def d2h_func(x, lda):
            d2h = lda[0] * np.array([[0.0, 0.0],
                                     [0.0, 0.0]])
            d2h += lda[0] * np.array([[0.0, 0.0],
                                      [0.0, 0.0]])
            d2h = csr_matrix(d2h)
            return(d2h)
        x0 =  np.random.randn(2).astype(float_dtype)
        p = IPFLS(x0=x0, 
                  f_func=f_func, df_func=df_func, d2f_func=d2f_func,
                  h_func=h_func, dh_func=dh_func, d2h_func=d2h_func,
                  float_dtype=float_dtype)
        x, s, lda, z, fval, signal = p.solve()
        print(x)
    elif prob == 5:
        print("maximize f(x,y,z) = -x*y*z subject to x+y+z = 1, x >= 0, y >= 0, z >= 0")
        def f_func(x):
            f = -np.array([x[0]*x[1]*x[2]])
            return f
        def df_func(x):
            df = -np.array([x[1]*x[2],
                            x[0]*x[2],
                            x[0]*x[1]])
            return df
        def d2f_func(x):
            d2f = -np.array([[0.0, x[2], x[1]],
                             [x[2], 0.0, x[0]],
                             [x[1], x[0], 0.0]])
            d2f = csr_matrix(d2f)
            return d2f
        def g_func(x):
            g = np.array([x[0] + x[1] + x[2] - 1.0])
            return g
        def dg_func(x):
            dg = np.array([[1.0],
                           [1.0],
                           [1.0]])
            return dg
        def d2g_func(x, lda):
            d2g = lda[0] * np.array([[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]])
            d2g = csr_matrix(d2g)
            return(d2g)
        x0 =  np.random.randn(3).astype(float_dtype)
        xmin = np.array([0,0,0]).astype(float_dtype)
#        A = np.array([1.0,1.0,1.0])
#        l = np.array([1.0])
#        u = np.array([1.0])+1e-10
        p = IPFLS(x0=x0, xmin=xmin, 
                  f_func=f_func, df_func=df_func, d2f_func=d2f_func,
                  g_func=g_func, dg_func=dg_func, d2g_func=d2g_func,
                  float_dtype=float_dtype)
        x, s, lda, z, fval, signal = p.solve()
        print(x)
    elif prob == 6:
        print("minimize f(x,y) = 100*(y-x**2)**2 + (1-x)**2")
        def f_func(x):
            f = 100.0 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
            return f
        
        def df_func(x):
            df = np.array([-400.0*(x[1]-x[0]**2)*x[0]-2*(1-x[0]),
                           200.0*(x[1]-x[0]**2)])
            return df
        
        def d2f_func(x):
            d2f = np.array([[-400.0*(x[1]-x[0]**2)+800.0*x[0]*x[0]+2.0, -400.0*x[0]],
                            [-400.0*x[0], 200.0]])
            d2f = csr_matrix(d2f)
            return d2f

        x0 =  np.array([0.0, 0.0])#np.random.randn(3).astype(float_dtype)
        p = IPFLS(x0=x0,
                  f_func=f_func, df_func=df_func, d2f_func=d2f_func,
                  float_dtype=float_dtype)
        x, s, lda, z, fval, signal = p.solve()
        print(x, fval)
    elif prob == 7:
        print("maximize f(x,y) = -x-y subject to x**2 + y**2 = 1")
        print("")
        x0 = np.random.randn(2).astype(float_dtype)
        def f_func(x):
            f = -np.array([x[0] + x[1]])
            return f
        
        def df_func(x):
            df = -np.array([1.0,
                            1.0])
            return df
        
        def d2f_func(x):
            d2f = -np.array([[0.0, 0.0],
                             [0.0, 0.0]])
            d2f = csr_matrix(d2f)
            return d2f
        
        def g_func(x):
            g = np.array([x[0]**2 + x[1]**2 - 1.0])
            return g
        
        def dg_func(x):
            dg = np.array([[2.0 * x[0]],
                           [2.0 * x[1]]])
            dg = csr_matrix(dg)
            return dg
        
        def d2g_func(x, lda):
            d2g = lda[0] * np.array([[2.0, 0.0],
                                     [0.0, 2.0]])
            d2g = csr_matrix(d2g)
            return(d2g)
        
        x0 =  np.random.randn(2).astype(float_dtype)
        p = IPFLS(x0=x0, #xmin=xmin, #xmax=xmax, 
                  f_func=f_func, df_func=df_func, d2f_func=d2f_func,
                  g_func=g_func, dg_func=dg_func, d2g_func=d2g_func,
    #              h_func=h_func, dh_func=dh_func, d2h_func=d2h_func,
                  float_dtype=float_dtype)
        x, s, lda, z, fval, signal = p.solve()
        print(x)
    elif prob == 8:
        print("maximize f(x,y) = - x1*x2 - x2*x3 \
               subject to x1**2 - x2**2 + x3**2 <= 2 \
                          x1**2 + x2**2 + x3**2 <= 10")
        print("")
        def f_func(x):
            f = -x[0]*x[1] - x[1]*x[2]
            return f
        
        def df_func(x):
            df = np.array([-x[1],
                           -x[0]-x[2],
                           -x[1]])
            return df
        
        def d2f_func(x):
            d2f = np.array([[0.0, -1.0, 0.0],
                            [-1.0, 0.0, -1.0],
                            [0.0, -1.0, 0.0]])
            d2f = csr_matrix(d2f)
            return d2f
        
        def h_func(x):
            h = np.array([x[0]**2 - x[1]**2 + x[2]**2 - 2.0,
                          x[0]**2 + x[1]**2 + x[2]**2 - 10.0])
            return h
        
        def dh_func(x):
            dh = np.array([[2.0 * x[0], 2.0 * x[0]],
                           [-2.0 * x[1], 2.0 * x[1]],
                           [2.0 * x[2], 2.0 * x[2]]
                          ])
            dh = csr_matrix(dh)
            return dh
        
        def d2h_func(x, lda):
            d2h = lda[0] * np.array([[2.0, 0.0, 0.0],
                                     [0.0, -2.0, 0.0],
                                     [0.0, 0.0, 2.0]
                                     ])
            d2h += lda[1] * np.array([[2.0, 0.0, 0.0],
                                      [0.0, 2.0, 0.0],
                                      [0.0, 0.0, 2.0]
                                      ])
            d2h = csr_matrix(d2h)
            return(d2h)
        
        x0 = np.array([1.0,1.0,0.0])
#        x0 = np.random.randn(3).astype(float_dtype)
        p = IPFLS(x0=x0, 
                  f_func=f_func, df_func=df_func, d2f_func=d2f_func,
                  h_func=h_func, dh_func=dh_func, d2h_func=d2h_func,
                  float_dtype=float_dtype)
        x, s, lda, z, fval, signal = p.solve()
        print(x)
        print(fval)
    elif prob == 9:
        print("minimize f(x,y,z) = 4*x-2*z subject to 2*x-y-z = 2, x**2 + y**2 = 1" )
        print("")
        def f_func(x):
            f = 4*x[0] - 2*x[2]
            return f
        def df_func(x):
            df = np.array([4.0, 0.0, -2.0])
            return df
        def d2f_func(x):
            d2f = np.array([[0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0]])
            d2f = csr_matrix(d2f)
            return d2f
        def g_func(x):
            g = np.array([2*x[0] - x[1] - x[2] - 2.0,
                          x[0]**2 + x[1]**2 - 1.0])
            return g
        def dg_func(x):
            dg = np.array([[2.0, 2*x[0]],
                           [-1.0, 2.0*x[1]],
                           [-1.0, 0.0]
                          ])
            dg = csr_matrix(dg)
            return dg
        def d2g_func(x, lda):
            d2g = lda[0] * np.array([[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]])
            d2g = d2g + lda[1] * np.array([[2.0, 0.0, 0.0],
                                           [0.0, 2.0, 0.0],
                                           [0.0, 0.0, 0.0]])
            d2g = csr_matrix(d2g)
            return(d2g)
        x0 = np.random.randn(3).astype(float_dtype)
        p = IPFLS(x0=x0,
                  f_func=f_func, df_func=df_func, d2f_func=d2f_func,
                  g_func=g_func, dg_func=dg_func, d2g_func=d2g_func,
                  float_dtype=float_dtype)
        x, s, lda, z, fval, signal = p.solve()
        print(x, fval)
    elif prob == 10:
        print("minimize f(x,y,z) = (x-1)**2 + 2*(y+2)**2 + 3*(z+3)**2 subject to z-y-x = 1, z-x**2 >= 0")
        def f_func(x):
            f = (x[0]-1.0)**2 + 2.0*(x[1]+2.0)**2 + 3.0*(x[2]+3.0)**2
            return f
        def df_func(x):
            df = np.array([2.0*(x[0]-1.0), 4.0*(x[1]+2.0), 6.0*(x[2]+3.0)])
            return df
        def d2f_func(x):
            d2f = np.array([[2.0, 0.0, 0.0],
                            [0.0, 4.0, 0.0],
                            [0.0, 0.0, 6.0]])
            d2f = csr_matrix(d2f)
            return d2f
        def g_func(x):
            g = np.array([-x[0]-x[1]+x[2]-1.0])
            return g
        def dg_func(x):
            dg = np.array([[-1.0],
                           [-1.0],
                           [1.0]])
            dg = csr_matrix(dg)
            return dg
        def d2g_func(x, lda):
            d2g = lda[0] * np.array([[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]])
            d2g = csr_matrix(d2g)
            return(d2g)
        def h_func(x):
            h = np.array([x[0]**2-x[2]])
            return h
        def dh_func(x):
            dh = np.array([[2.0*x[0]],
                           [0.0],
                           [-1.0]])
            dh = csr_matrix(dh)
            return dh
        def d2h_func(x, lda):
            d2h = lda[0] * np.array([[2.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]])
            d2h = csr_matrix(d2h)
            return(d2h)
        x0 = np.random.randn(3).astype(float_dtype)
        p = IPFLS(x0=x0,
                  f_func=f_func, df_func=df_func, d2f_func=d2f_func,
                  g_func=g_func, dg_func=dg_func, d2g_func=d2g_func,
                  h_func=h_func, dh_func=dh_func, d2h_func=d2h_func,
                  float_dtype=float_dtype)
        x, s, lda, z, fval, signal = p.solve()
        print(x)
    elif prob == 11:
        print("minimize f(x,y,z) = (x-1)**2 + 2*(y+2)**2 + 3*(z+3)**2 subject to z-y-x = 1, z-x**2 >= 0")
        def f_func(x, a):
            f = (x[0]-a[0])**2 + 2.0*(x[1]+a[1])**2 + 3.0*(x[2]+a[2])**2
            return f
        def df_func(x):
            df = np.array([2.0*(x[0]-1.0), 4.0*(x[1]+2.0), 6.0*(x[2]+3.0)])
            return df
        def d2f_func(x):
            d2f = np.array([[2.0, 0.0, 0.0],
                            [0.0, 4.0, 0.0],
                            [0.0, 0.0, 6.0]])
            d2f = csr_matrix(d2f)
            return d2f
        def g_func(x):
            g = np.array([-x[0]-x[1]+x[2]-1.0])
            return g
        def dg_func(x):
            dg = np.array([[-1.0],
                           [-1.0],
                           [1.0]])
            dg = csr_matrix(dg)
            return dg
        def d2g_func(x, lda):
            d2g = lda[0] * np.array([[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]])
            d2g = csr_matrix(d2g)
            return(d2g)
        def h_func(x):
            h = np.array([x[0]**2-x[2]])
            return h
        def dh_func(x):
            dh = np.array([[2.0*x[0]],
                           [0.0],
                           [-1.0]])
            dh = csr_matrix(dh)
            return dh
        def d2h_func(x, lda):
            d2h = lda[0] * np.array([[2.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]])
            d2h = csr_matrix(d2h)
            return(d2h)
        
        a = np.array([1.0, 2.0, 3.0])
        f_fcn = lambda x: f_func(x, a)
        x0 = np.random.randn(3).astype(float_dtype)
        p = IPFLS(x0=x0,
                  f_func=f_fcn, df_func=df_func, d2f_func=d2f_func,
                  g_func=g_func, dg_func=dg_func, d2g_func=d2g_func,
                  h_func=h_func, dh_func=dh_func, d2h_func=d2h_func,
                  float_dtype=float_dtype)
        x, s, lda, z, fval, signal = p.solve()
        print(x)

if __name__ == '__main__':
    main()
