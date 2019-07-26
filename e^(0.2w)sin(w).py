from sys import path
path.append(r"d:/casadi-windows-py37-58aa427")
#path.append(r"H:/Python/casadi-windows-py37-58aa427")
from casadi import *
import matplotlib.pyplot as plt

x=SX.sym('w') # Decision variables (controls)
obj=exp(0.2*x)*sin(x) #obj
print(obj)

xx=np.linspace(0,14)
yy=exp(0.2*xx)*sin(xx)
plt.plot(xx, yy, "b")
plt.show()


g=[] # Optimization constraints – empty (unconstrained)
P=[] # Optimization problem parameters – empty (no parameters used here)
OPT_variables=x
nlp_prob={'f':obj,'x':OPT_variables,'g':g,'p':P}

opts={}
opts['print_time']=False
opts['ipopt']={'max_iter':100,'print_level':0,'acceptable_tol':1e-8,'acceptable_obj_change_tol' : 1e-6}
print(opts)

solver = nlpsol("solver", "ipopt", nlp_prob,opts)

nl = NlpBuilder()

nl.lbx=0 # constrained optimization: w>=0 w<=4pi
nl.ubx=4*pi
# nl.lbx=-3
# nl.ubx=1

nl.lbg=-inf
nl.ubg=inf

#optimal value depends on the initial value x0
nl.p=[]
nl.x0=4
# #input of solver can also be given as args = dict(lbx=-inf, ubx=inf, lbg=-inf, ubg=inf,  # unconstrained optimization
#             p=[],  # no parameters
#             x0=-0.5  # initial guess
#            )
# sol=solver(**args)
sol = solver(lbx=nl.lbx,
             ubx=nl.ubx,
             lbg=nl.lbg,
             ubg=nl.ubg,
             p=nl.p,
             x0=nl.x0)
print(sol)
print('at x=%f get optimal value' % sol['x'])
print('optimal value is: f=%f' % sol['f'])