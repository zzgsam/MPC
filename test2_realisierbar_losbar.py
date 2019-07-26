from ipydex import IPS, activate_ips_on_exception
from sys import path
# path.append(r"d:/casadi-windows-py37-58aa427")
# path.append(r"H:/Python/casadi-windows-py37-58aa427")


from casadi import *
import matplotlib.pyplot as plt

N_pred =5

state_aktuell=[]
# activate_ips_on_exception()

# IPS()


x = SX.sym('x')
u = SX.sym('u')

states = vertcat(x)
# print(states.shape)
n_states = states.shape[0]  # p 22.

controls = vertcat(u)
n_controls = controls.shape[0]
rhs = vertcat(x+u)

# system r.h.s eta_p h_p h_pp
print(rhs)

# nonlinear mapping function f(x,u)
f = Function('f', [states, controls], [rhs])
U = SX.sym('U', n_controls, N_pred)  # Decision variables (controls) u0 u1 u2
# parameters (which include the !!!initial and the !!!reference state of
# the robot)
P = SX.sym('P', n_states)
X = SX.sym('X', n_states, (N_pred + 1))  # number of x is always N+1   -> 0..N
print("X gerade definiert",X)
X[:, 0] = P[0]
obj = 0
g = []  # constraints vector
Q = SX.zeros(1, 1)
Q[0, 0] = 1



R = SX.zeros(1, 1)
R[0, 0] = 1

# compute predicted states
for k in range(N_pred):
    st = X[:, k]
    con = U[:, k]
    f_value = f(st, con)
   # print(f_value)
    st_next = f_value
    X[:, k + 1] = st_next
    obj = obj + (st).T@Q@(st ) + (con).T@R@(con)
print("X danach", X)
print(obj)

#g = SX.zeros(2 * (N_pred + 1), 1)  # create a dense matrix
g = SX.zeros((N_pred + 1), 1)  # create a dense matrix

print(g.size())
for k in range(N_pred + 1):
    g[k] = X[0, k]

print("g",g)

OPT_variables = reshape(U, N_pred, 1)

nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}
opts = {}
opts['print_time'] = False
opts['ipopt'] = {
    'max_iter': 200,
    'print_level': 0,
    'acceptable_tol': 1e-8,
    # 'fixed_variable_treatment':'make_constraint',
    'acceptable_constr_viol_tol':1e-4,
    'print_info_string':'no',
    'expect_infeasible_problem':'yes',
    'acceptable_obj_change_tol': 1e-20}
# print(opts)
solver = nlpsol("solver", "ipopt", nlp_prob, opts)
print(solver)


nl = {}

nl['lbx'] = [0 for i in range(N_pred)]
nl['ubx'] = [0 for i in range(N_pred)]
# ??? better method?
for i in range(N_pred):
    nl['lbx'][i] = -1
    nl['ubx'][i] = 1
# multi-constraints defined by list
# initialisation
nl['lbg'] = [-inf for i in range( (N_pred + 1))]
nl['ubg'] = [inf for i in range( (N_pred + 1))]
nl['lbg'][-1]=0
nl['ubg'][-1]=0
#
print(nl['ubg'])

state_aktuell.append(4.3)
input_aktuell=[]
nl['p'] = [4.3]
nl['x0'] = [0 for i in range(N_pred)]

sol = solver(**nl)
print(sol)

for i in range(N_pred):
    state_aktuell.append(float(sol['g'][1]))
    input_aktuell.append(float(sol['x'][0]))
    nl['p']=float(sol['g'][1])
    sol = solver(**nl)
    print(sol)
print(state_aktuell,input_aktuell)