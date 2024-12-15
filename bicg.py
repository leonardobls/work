import numpy as np
import sys
import profile
import pstats

IMAX = 100000
ERRO = 1e-6

#--------------------------------------------------------------------------
def bicg(A, x):

    nlin, ncols = A.shape
    x = np.zeros((nlin,1))
    r = np.subtract(b,np.matmul(A,x))
    r2 = r
    p = np.zeros((nlin,1))
    p2 = np.zeros((nlin,1))
    rho = 1

    i = 1

    while  i < IMAX:
        rho0 = rho
        rho = np.dot(np.transpose(r2), r)
        beta = rho / rho0
        p = np.add(r,np.multiply(beta,p))
        p2 = np.add(r2,np.multiply(beta,p2))
        v = np.matmul(A,p)
        alpha = rho/np.dot(np.transpose(p2), v)
        x = np.add(x,np.multiply(alpha,p))

        if np.dot(np.transpose(r), r) < ERRO * ERRO: 
            break

        r = np.subtract(r, np.multiply(alpha,v))
        r2 = np.subtract(r2, np.multiply(alpha, np.matmul(np.transpose(A),p2)))
        i = i + 1

    return x, i
#--------------------------------------------------------------------------
if len(sys.argv)!=2:
    print('%s < ordem da matriz>' % sys.argv[0])
    sys.exit(0)

n = int(sys.argv[1]) 

A = np.random.rand(n,n)
b = np.ones((n,1))

x, niter = bicg(A,b) 
print('Iteracoes: %d' % niter)

b = np.matmul(A,x)
print(b)



