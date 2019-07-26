from ipydex import IPS, activate_ips_on_exception
from sys import path
# path.append(r"d:/casadi-windows-py37-58aa427")
# path.append(r"H:/Python/casadi-windows-py37-58aa427")


from casadi import *
import matplotlib.pyplot as plt

A_B = 2.8274e-3  # [m**2]
A_SP = 0.4299e-3  # [m**2]
m = 2.8e-3  # [kg]
g = 9.81  # [m/(s**2)]
T_M = 0.57  # [s]
k_M = 0.31  # [s**-1]
k_V = 6e-5  # [m**3]
k_L = 2.18e-4  # [kg/m]
eta_0 = 1900 / 60 # / 60 * 2 * pi
#
h_ub = 2
h_lb = 0
eta_ub = 200
eta_lb = 0
# h_ub = inf
# h_lb = -inf
# eta_ub = inf
# eta_lb = -inf


T = 0.1
N_pred =19

input_r = [157.29]
# activate_ips_on_exception()

# IPS()


h = SX.sym('h')
h_p = SX.sym('h_p')
eta = SX.sym('eta')

states = vertcat(h, h_p, eta)
# print(states.shape)
n_states = states.shape[0]  # p 22.


u_pwm = SX.sym('u_pwm')
controls = vertcat(u_pwm)
n_controls = controls.shape[0]
rhs = vertcat(h_p, k_L / m * ((k_V * (eta + eta_0) - A_B * h_p) /
                               A_SP)**2 - g, -1 / T_M * eta + k_M / T_M * u_pwm)

# system r.h.s eta_p h_p h_pp
print(rhs)

# nonlinear mapping function f(x,u)
f = Function('f', [states, controls], [rhs])
U = SX.sym('U', n_controls, N_pred)  # Decision variables (controls) u0 u1 u2
# parameters (which include the !!!initial and the !!!reference state of
# the robot)
P = SX.sym('P', n_states + n_states)
X = SX.sym('X', n_states, (N_pred + 1))  # number of x is always N+1   -> 0..N

X[:, 0] = P[0:3]
obj = 0
g = []  # constraints vector
Q = SX.zeros(3, 3)
Q[0, 0] = 1
Q[1, 1] = 1
Q[2, 2] = 1

R = SX.zeros(1, 1)
R[0, 0] = 1

# compute predicted states
for k in range(N_pred):
    st = X[:, k]
    con = U[:, k]
    f_value = f(st, con)
   # print(f_value)
    st_next = st + (T * f_value)
    X[:, k + 1] = st_next
    obj = obj + (st - P[3:6]).T@Q@(st - P[3:6]) + (con-input_r).T@R@(con-input_r)
print(obj)

#g = SX.zeros(2 * (N_pred + 1), 1)  # create a dense matrix
g = SX.zeros(3 * (N_pred + 1), 1)  # create a dense matrix

print(g.size())
for k in range(N_pred + 1):
    # h
    g[3 * k] = X[0, k]
#    print(g[k])

    g[3 * k + 1] = X[1, k]
#    print(g[k+1])
    # eta
    g[3 * k + 2] = X[2, k]
print(g)

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
    nl['lbx'][i] = 0
    nl['ubx'][i] = 255
# multi-constraints defined by list
# initialisation
nl['lbg'] = [0 for i in range(3 * (N_pred + 1))]
nl['ubg'] = [0 for i in range(3 * (N_pred + 1))]
#
nl['lbg'][::3] = [h_lb] * (N_pred + 1)
nl['ubg'][::3] = [h_ub] * (N_pred + 1)
nl['lbg'][1::3] = [-inf] * (N_pred + 1)
nl['ubg'][1::3] = [inf] * (N_pred + 1)
nl['lbg'][2::3] = [eta_lb] * (N_pred + 1)
nl['ubg'][2::3] = [eta_ub] * (N_pred + 1)
# nl['lbg'][-2] = 2
# nl['ubg'][-2] = 2
nl['lbg'][-3] = 0.8
nl['ubg'][-3] = 0.8
# nl['lbg'][-1] = 55
# nl['ubg'][-1] = 55
print(nl['ubg'])


nl['p'] = [0, 0, 40, 0.8, 0, 48]
nl['x0'] = [0 for i in range(N_pred)]

sol = solver(**nl)

# print(type(float((sol['g'][2]))))
print(solver.stats())
print(sol)
print((solver.stats())['success']==True)
print(sol['x'].shape)
eta_out = []
h_out = []
t = [T * i for i in range(N_pred+1)]

for i in range(N_pred+1):
    h_out.append(float(sol['g'][3 * i]))  # extend
    #eta_out.append(float(sol['g'][3 * i + 1]))
#plt.plot(t, eta_out, "r")
plt.plot(t, h_out, "b")
plt.xlabel('t')
#plt.ylabel('eta,h')
plt.ylabel('h')

plt.show()

print(f(sol['g'][-6:-3],sol['x'][-1])*T+sol['g'][-6:-3])