import numpy as np
import matplotlib
from matplotlib  import pyplot as plt
import time
from numba import njit
matplotlib.rcParams.update({'font.size': 22})

# Parameter settings
a = 1
b = 2

n1 = 128
h1 = 1./n1
dt1 = h1/10
nt1 = int(10/dt1)
mu1 = a*dt1/h1
nu1 = b*dt1/h1

n2 = 256
h2 = 1./n2
dt2 = h2/10
nt2 = int(10/dt2)
mu2 = a*dt2/h2
nu2 = b*dt2/h2

n3 = 512
h3 = 1./n3
dt3 = h3/10
nt3 = int(10/dt3)
mu3 = a*dt3/h3
nu3 = b*dt3/h3

# N=128
# Initial conditions
uold1 = np.zeros([n1+3,n1+3])
unew1 = np.zeros([n1+3,n1+3])

x1 = np.zeros([n1+1,n1+1])
y1 = np.zeros([n1+1,n1+1])

for i in range(n1+1):
    x1[i]   = np.linspace(0, 1.0, n1+1)
    y1[:,i] = np.linspace(0, 1.0, n1+1)

uold1[1:-1, 1:-1] = np.exp(- (np.power(x1-0.5,2) + np.power(y1-0.5,2)) / 0.15**2)
#print(uold)
uold1[0] = uold1[-2]
uold1[-1] = uold1[1]
uold1[:,0] = uold1[:,-2]
uold1[:,-1] = uold1[:,1]
#print(uold)

err1 = np.zeros(nt1+1)
t1 = np.linspace(0, 10, nt1+1)
start = time.time()

for i in range(nt1):
    # Lax-Wendroff solution
    unew1[1:-1,1:-1] = (1-mu1**2-nu1**2+mu1*nu1)*uold1[1:-1,1:-1] + mu1*nu1*uold1[:-2,:-2] \
    + mu1*(mu1-1)/2*uold1[2:,1:-1] + nu1*(nu1-1)/2*uold1[1:-1,2:] \
    + mu1*(0.5+0.5*mu1-nu1)*uold1[:-2,1:-1] + nu1*(0.5+0.5*nu1-mu1)*uold1[1:-1,:-2] 
    
    # Boundary Condition
    unew1[0,1:-1] = unew1[-2,1:-1]
    unew1[-1,1:-1] = unew1[1,1:-1]
    unew1[1:-1,0] = unew1[1:-1,-2]
    unew1[1:-1,-1] = unew1[1:-1,1]
  
    # Update the result
    uold1[:,:] = unew1[:,:]
    
    # Analytical solution
    x = np.remainder(x1-a*t1[i+1], 1.0)
    y = np.remainder(y1-b*t1[i+1], 1.0)
    uana = np.exp(- (np.power(x-0.5,2) + np.power(y-0.5,2)) / 0.15**2)
    #uana = np.exp(- (np.power((x1-a*t1[i+1])-0.5,2) + np.power((y1-b*t1[i+1])-0.5,2)) / 0.15**2)
    err1[i+1] = np.sqrt(np.sum(np.power(unew1[1:-1,1:-1]-uana,2))/(n1+1)**2)
    
    if(((i+1) % 1000)==0):
        print(i+1,err1[i+1])

end = time.time()
print("time elapsed=",end-start)

# N=256
# Initial conditions
uold2 = np.zeros([n2+3,n2+3])
unew2 = np.zeros([n2+3,n2+3])

x2 = np.zeros([n2+1,n2+1])
y2 = np.zeros([n2+1,n2+1])

for i in range(n2+1):
    x2[i]   = np.linspace(0, 1.0, n2+1)
    y2[:,i] = np.linspace(0, 1.0, n2+1)

uold2[1:-1, 1:-1] = np.exp(- (np.power(x2-0.5,2) + np.power(y2-0.5,2)) / 0.15**2)
#print(uold)
uold2[0] = uold2[-2]
uold2[-1] = uold2[1]
uold2[:,0] = uold2[:,-2]
uold2[:,-1] = uold2[:,1]
#print(uold)

err2 = np.zeros(nt2+1)
t2 = np.linspace(0, 10, nt2+1)
start = time.time()

for i in range(nt2):
    # Lax-Wendroff solution
    unew2[1:-1,1:-1] = (1-mu2**2-nu2**2+mu2*nu2)*uold2[1:-1,1:-1] + mu2*nu2*uold2[:-2,:-2] \
    + mu2*(mu2-1)/2*uold2[2:,1:-1] + nu2*(nu2-1)/2*uold2[1:-1,2:] \
    + mu2*(0.5+0.5*mu2-nu2)*uold2[:-2,1:-1] + nu2*(0.5+0.5*nu2-mu2)*uold2[1:-1,:-2] 
    
    # Boundary Condition
    unew2[0,1:-1] = unew2[-2,1:-1]
    unew2[-1,1:-1] = unew2[1,1:-1]
    unew2[1:-1,0] = unew2[1:-1,-2]
    unew2[1:-1,-1] = unew2[1:-1,1]
  
    # Update the result
    uold2[:,:] = unew2[:,:]
    
    # Analytical solution
    x = np.remainder(x2-a*t2[i+1], 1.0)
    y = np.remainder(y2-b*t2[i+1], 1.0)
    uana = np.exp(- (np.power(x-0.5,2) + np.power(y-0.5,2)) / 0.15**2)
    #uana = np.exp(- (np.power((x2-a*t2[i+1])-0.5,2) + np.power((y2-b*t2[i+1])-0.5,2)) / 0.15**2)
    err2[i+1] = np.sqrt(np.sum(np.power(unew2[1:-1,1:-1]-uana,2))/(n2+1)**2)
    
    if(((i+1) % 1000)==0):
        print(i+1,err2[i+1])

end = time.time()
print("time elapsed=",end-start)

# N=512
# Initial conditions
uold3 = np.zeros([n3+3,n3+3])
unew3 = np.zeros([n3+3,n3+3])

x3 = np.zeros([n3+1,n3+1])
y3 = np.zeros([n3+1,n3+1])

for i in range(n3+1):
    x3[i]   = np.linspace(0, 1.0, n3+1)
    y3[:,i] = np.linspace(0, 1.0, n3+1)

uold3[1:-1, 1:-1] = np.exp(- (np.power(x3-0.5,2) + np.power(y3-0.5,2)) / 0.15**2)
#print(uold)
uold3[0] = uold3[-2]
uold3[-1] = uold3[1]
uold3[:,0] = uold3[:,-2]
uold3[:,-1] = uold3[:,1]
#print(uold)

err3 = np.zeros(nt3+1)
t3 = np.linspace(0, 10, nt3+1)
start = time.time()

for i in range(nt3):
    # Lax-Wendroff solution
    unew3[1:-1,1:-1] = (1-mu3**2-nu3**2+mu3*nu3)*uold3[1:-1,1:-1] + mu3*nu3*uold3[:-2,:-2] \
    + mu3*(mu3-1)/2*uold3[2:,1:-1] + nu3*(nu3-1)/2*uold3[1:-1,2:] \
    + mu3*(0.5+0.5*mu3-nu3)*uold3[:-2,1:-1] + nu3*(0.5+0.5*nu3-mu3)*uold3[1:-1,:-2] 
    
    # Boundary Condition
    unew3[0,1:-1] = unew3[-2,1:-1]
    unew3[-1,1:-1] = unew3[1,1:-1]
    unew3[1:-1,0] = unew3[1:-1,-2]
    unew3[1:-1,-1] = unew3[1:-1,1]
  
    # Update the result
    uold3[:,:] = unew3[:,:]
    
    # Analytical solution
    x = np.remainder(x3-a*t3[i+1], 1.0)
    y = np.remainder(y3-b*t3[i+1], 1.0)
    uana = np.exp(- (np.power(x-0.5,2) + np.power(y-0.5,2)) / 0.15**2)
    #uana = np.exp(- (np.power((x3-a*t3[i+1])-0.5,2) + np.power((y3-b*t3[i+1])-0.5,2)) / 0.15**2)
    err3[i+1] = np.sqrt(np.sum(np.power(unew3[1:-1,1:-1]-uana,2))/(n3+1)**2)
    
    if(((i+1) % 1000)==0):
        print(i+1,err3[i+1])

end = time.time()
print("time elapsed=",end-start)

# Plot the L2 error
plt.plot(t1[1:], err1[1:], label='N=128')
plt.plot(t2[1:], err2[1:], label='N=256')
plt.plot(t3[1:], err3[1:], label='N=512')
plt.xlabel('time')
plt.ylabel(r'$\Vert e\Vert_{L^2}$')
plt.title(r'Errors of Lax-Wendroff')
plt.legend()
plt.grid()
plt.savefig('p3a_2.png')

# Plot the final state
plt.figure(figsize=[6.4, 15])
plt.subplot(311)
Cont1 = plt.contourf(x1, y1, unew1[1:-1,1:-1])
Cbar1 = plt.colorbar(Cont1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('u(10,x,y), LW, N=128')

plt.subplot(312)
Cont2 = plt.contourf(x2, y2, unew2[1:-1,1:-1])
Cbar2 = plt.colorbar(Cont2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('u(10,x,y), LW, N=256')

plt.subplot(313)
Cont3 = plt.contourf(x3, y3, unew3[1:-1,1:-1])
Cbar3 = plt.colorbar(Cont3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('u(10,x,y), LW, N=512')

plt.tight_layout()
plt.savefig('p3b_2.png')
