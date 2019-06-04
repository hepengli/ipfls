# ipopt
This is a Python implemetationa of the IPOPT optimization toolbox using a prime-dual interior-point algorithm with a filter line-search method for nonlinear programming. The problem is supposed to formulate in the following expression:

        min f(x)
         x

        subject to:

        g(x) = 0            (nonlinear equalities)
        h(x) <= 0           (nonlinear inequalities)
        l <= A*x <= u       (linear constraints)
        xmin <= x <= xmax   (variable bounds)
