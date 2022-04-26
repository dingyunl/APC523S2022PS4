import numpy as np
import matplotlib
from matplotlib  import pyplot as plt
matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.rcParams['lines.markeredgewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 4
matplotlib.rcParams['font.size'] = 18

def jacobian(u,n):
    
    h = 1 / n
    
    a = np.zeros([(n-1)*(n-1), (n-1)*(n-1)])
    for i in range(1,n-2):
        for j in range(1,n-2):
            a[i+j*(n-1),i+j*(n-1)] = - 4 - 4*h**2*u[i+j*(n-1)]**3
            a[i+j*(n-1),i-1+j*(n-1)]   = 1
            a[i+j*(n-1),i+1+j*(n-1)]   = 1
            a[i+j*(n-1),i+(j-1)*(n-1)] = 1
            a[i+j*(n-1),i+(j+1)*(n-1)] = 1
            
    for k in range(1,n-2):
        a[k,k]       = - 4 - 4*h**2*u[k]**3
        a[k,k-1]     = 1
        a[k,k+1]     = 1
        a[k,k+(n-1)] = 1
        a[k+(n-2)*(n-1),k+(n-2)*(n-1)]     = - 4 - 4*h**2*u[k+(n-2)*(n-1)]**3
        a[k+(n-2)*(n-1),k-1+(n-2)*(n-1)]   = 1
        a[k+(n-2)*(n-1),k+1+(n-2)*(n-1)]   = 1
        a[k+(n-2)*(n-1),k+(n-3)*(n-1)]     = 1
    
        a[k*(n-1),k*(n-1)]     = - 4 - 4*h**2*u[k*(n-1)]**3
        a[k*(n-1),1+k*(n-1)]   = 1
        a[k*(n-1),(k-1)*(n-1)] = 1
        a[k*(n-1),(k+1)*(n-1)] = 1
        a[n-2+k*(n-1),n-2+k*(n-1)]     = - 4 - 4*h**2*u[n-2+k*(n-1)]**3
        a[n-2+k*(n-1),n-3+k*(n-1)]     = 1
        a[n-2+k*(n-1),n-2+(k-1)*(n-1)] = 1
        a[n-2+k*(n-1),n-2+(k+1)*(n-1)] = 1 

    a[0,0]   = -4-4*h**2*u[0]**3
    a[0,1]   = 1
    a[0,n-1] = 1
    a[n-2,n-2]       = -4-4*h**2*u[n-2]**3
    a[n-2,n-3]       = 1
    a[n-2,n-2+(n-1)] = 1
    a[(n-2)*(n-1),(n-2)*(n-1)]   = -4-4*h**2*u[(n-2)*(n-1)]**3
    a[(n-2)*(n-1),1+(n-2)*(n-1)] = 1
    a[(n-2)*(n-1),(n-3)*(n-1)]   = 1
    a[n-2+(n-2)*(n-1), n-2+(n-2)*(n-1)] = -4-4*h**2*u[n-2+(n-2)*(n-1)]**3
    a[n-2+(n-2)*(n-1), n-3+(n-2)*(n-1)] = 1 
    a[n-2+(n-2)*(n-1), n-2+(n-3)*(n-1)] = 1

    return a/h**2

def au(u,n):
    
    h = 1/n
    
    au = np.zeros((n-1)*(n-1))
    for i in range(1,n-2):
        for j in range(1,n-2):
            au[i+j*(n-1)] = (u[i+1+j*(n-1)]+u[i-1+j*(n-1)]+u[i+(j+1)*(n-1)]+u[i+(j-1)*(n-1)]
                             -4*u[i+j*(n-1)])/h**2 - u[i+j*(n-1)]**4

    for k in range(1,n-2):
        au[k]             = (u[k+1]+u[k-1]+u[k+(n-1)]+1-4*u[k])/h**2 - u[k]**4
        au[k+(n-2)*(n-1)] = (u[k+1+(n-2)*(n-1)]+u[k-1+(n-2)*(n-1)]+1+u[k+(n-3)*(n-1)]
                             -4*u[k+(n-2)*(n-1)])/h**2 - u[k+(n-2)*(n-1)]**4
        au[k*(n-1)]       = (u[k*(n-1)+1]+1+u[(k+1)*(n-1)]+u[(k-1)*(n-1)]
                             -4*u[k*(n-1)])/h**2 - u[k*(n-1)]**4
                             
        
        au[n-2+k*(n-1)]   = (1+u[n-3+k*(n-1)]+u[n-2+(k+1)*(n-1)]+u[n-2+(k-1)*(n-1)]
                            -4*u[n-2+k*(n-1)])/h**2 - u[n-2+k*(n-1)]**4

    au[0]              = (2+u[1]+u[n-1]-4*u[0])/h**2 - u[0]**4
    au[n-2]            = (2+u[n-3]+u[n-2+(n-1)]-4*u[n-2])/h**2 - u[n-2]**4
    au[(n-2)*(n-1)]    = (2+u[(n-2)*(n-1)+1]+u[(n-3)*(n-1)]
                          -4*u[(n-2)*(n-1)])/h**2 - u[(n-2)*(n-1)]**4
    au[n-2+(n-2)*(n-1)] = (2+u[n-3+(n-2)*(n-1)]+u[n-2+(n-3)*(n-1)]
                          -4*u[n-2+(n-2)*(n-1)])/h**2 - u[n-2+(n-2)*(n-1)]**4
    return au

def bvp_nonlinear_2D(n,plot=False):
    
    h = 1/n
    
    # initial guess with correct boundary conditions
    u = np.ones((n-1)*(n-1))
    
    error = 1
    iter = 0
    while (error > 1e-10):
        '''
        if (plot):
            plt.plot(x,u,"o-")
        '''   
        J = jacobian(u,n)
        du = -np.matmul(np.linalg.inv(J), au(u,n))

        u += du
        
        error = np.max(abs(du))
        iter = iter + 1
        if(plot):
            print(iter,error)
    return u

N = 64
L = 1
x = np.linspace(0, L, N+1)
u = bvp_nonlinear_2D(N, False)

uu = np.zeros([N+1,N+1])
uu[0]     = 1
uu[N]     = 1
uu[:,0]   = 1
uu[:,N] = 1
for i in range(1, N):
    uu[i, 1:-1] = u[(i-1)*(N-1):i*(N-1)]        

plt.figure(figsize=(8,8))
Cont = plt.contourf(x,x,uu)
Cbar = plt.colorbar(Cont)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Direct matrix inversion for N=64')
plt.savefig('p1_a.png')
