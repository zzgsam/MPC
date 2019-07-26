from sys import path
path.append(r"d:/casadi-windows-py37-58aa427")
from casadi import *
from inspect import signature
# x = MX.sym("x")
# print(jacobian(sin(x),x))
#CASADI API: https://web.casadi.org/python-api/#nlp
#
#
# Remarks:
# - Single optimization variable
# - Unconstrained optimization
# - Local minimum = Global minimum
x=MX.sym('w')
obj=x**2-6*x+13
print('obj:',obj)

g=[]
P=[]

OPT_variables=x
nlp_prob={"f":obj, "x":OPT_variables, "g":g, "p": P}
print(nlp_prob)

nl = NlpBuilder()


nl.lbx=-inf
nl.ubx=inf
# nl.lbx=-3
# nl.ubx=1

nl.lbg=-inf
nl.ubg=inf
print(nl)

opts={}
opts["expand"] = True
#opts["max_iter"] = 10
# opts["verbose"] = True
# opts["linear_solver"] = "ma57"
# opts["hessian_approximation"] = "limited-memory"

solver = nlpsol("solver", "ipopt", nlp_prob,opts)


#solver.print_options()
nl.p=[]
nl.x0=-0.5
sol = solver(lbx=nl.lbx,
             ubx=nl.ubx,
             lbg=nl.lbg,
             ubg=nl.ubg,
             p=nl.p,
             x0=nl.x0)
print(sol)
print('at x=%d get optimal value' % sol['x'])
print('optimal value is: f=%d' % sol['f'])


