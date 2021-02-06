# IPFLS

This is a Python implemetationa of the prime-dual interior-point algorithm with a filter line-search method [1] for large-scale nonlinear programming. The problem is supposed to formulate in the following expression:

        min f(x)
         x

        subject to:

        g(x) = 0            (nonlinear equalities)
        h(x) <= 0           (nonlinear inequalities)
        l <= A*x <= u       (linear constraints)
        xmin <= x <= xmax   (variable bounds)

[1] Wächter, A., Biegler, L. On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. Math. Program. 106, 25–57 (2006). https://doi.org/10.1007/s10107-004-0559-y
