import numpy as np
import matplotlib
from matplotlib  import pyplot as plt
import time
from numba import njit
matplotlib.rcParams.update({'font.size': 18})

n = 512
h = 1./n
eps = 1e-15
itermax = 20000
bc = 'dirichlet'

uold = np.ones([n+1,n+1])
unew = np.zeros([n+1,n+1])
omega = 2/3
convj = np.zeros(itermax)

iter = 0
norm = 1
frac = h**2
start = time.time()

while (iter < itermax and norm > eps):
    
    unew[1:-1,1:-1] = uold[1:-1,1:-1] + 0.25 * omega * \
    (uold[2:,1:-1]+uold[:-2,1:-1]+uold[1:-1,2:]+uold[1:-1,:-2]-4.0*uold[1:-1,1:-1] 
     - frac * np.power(uold[1:-1,1:-1],4))
            
    if(bc=='dirichlet'):
        unew[0] = 1.0 
        unew[-1] = 1.0
        unew[:,0] = 1.0 
        unew[:,-1] = 1.0
    if(bc=='neumann'):
        unew[0,1:-1] = unew[1,1:-1]
        unew[-1,1:-1] = unew[-2,1:-1]
        unew[1:-1,0] = unew[1:-1,1]
        unew[1:-1,-1] = unew[1:-1,-2]
    if(bc=='periodic'):
        unew[0,1:-1] = unew[-2,1:-1]
        unew[-1,1:-1] = unew[1,1:-1]
        unew[1:-1,0] = unew[1:-1,-2]
        unew[1:-1,-1] = unew[1:-1,1]

    norm = np.max(abs(unew - uold))

    convj[iter] = norm
    
    uold[:,:] = unew[:,:]
    
    iter = iter + 1
    if((iter % 1000)==0):
        print(iter,norm)

print(iter,norm)
end = time.time()
print("time elapsed=",end-start)

x = np.linspace(0, 1.0, n+1)
plt.figure(figsize=(8,8))
Cont = plt.contourf(x,x,unew)
Cbar = plt.colorbar(Cont)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Jacobi Iterative Solver for N='+str(n))
plt.savefig('p1_b_'+str(n)+'.png')
