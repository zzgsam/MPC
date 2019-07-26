from sys import path
path.append(r"d:/casadi-windows-py37-58aa427")
#path.append(r"H:/Python/casadi-windows-py37-58aa427")
from casadi import *
import matplotlib.pyplot as plt



x=[0,45,90,135,180]
y=[667,661,757,871,1210]

plt.plot(x,y,"b*")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
#plt.show()

m=SX.sym('m')
c=SX.sym('c')

obj=0

#for X, Y in zip(XX, YY):
for i in range(len(x)):
    obj += (y[i]-(m*x[i]+c))**2
print(obj)
g=[]
P=[]

OPT_variables=SX(2,1)
OPT_variables[0,0]=m
OPT_variables[1,0]=c

nlp_prob={'f':obj,'x':OPT_variables,'g':g,'p':P}

opts={}
opts['print_time']=False
opts['ipopt']={'max_iter':1000,'print_level':0,'acceptable_tol':1e-8,'acceptable_obj_change_tol' : 1e-6}

solver = nlpsol("solver", "ipopt", nlp_prob,opts)

nl = NlpBuilder()
nl.lbx=-inf
nl.ubx=inf
nl.lbg=-inf
nl.ubg=inf
nl.p=[]
nl.x0=[0.5,1]

# sol=solver(**args)
sol = solver(lbx=nl.lbx,
             ubx=nl.ubx,
             lbg=nl.lbg,
             ubg=nl.ubg,
             p=nl.p,
             x0=nl.x0)
print(sol)
print('at m=%f,c=%f get optimal value' % (sol['x'][0],sol['x'][1]))
print('optimal value is: f=%f' % sol['f'])

#np.linspace(0,180)
x_line=range(0,180)

m_sol=sol['x'][0]
c_sol=sol['x'][1]
y_line=m_sol*x_line+c_sol
plt.plot(x_line,y_line,"r")
plt.xlabel('x')
plt.ylabel('y')

plt.show()

#objective visaulization
#import casadi as cs
#cs.Function
obj_fun=Function('obj_fun',[m,c],[obj])

m_range=np.arange(-1,6,0.5)
c_range=np.arange(400,800,50)
obj_plot_data=[]
print(m_range)

mm,cc=np.meshgrid(m_range,c_range)
obj_plot_data = np.array(obj_fun(mm, cc))

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1,1,1, projection='3d')
#ax.plot
ax.plot_surface(mm, cc, obj_plot_data, cmap=cmx.viridis) #cmap=plt.cm.hot
ax.set_xlabel("$m$")
ax.set_ylabel("$c$")
ax.set_zlabel("$\phi$")
plt.show()