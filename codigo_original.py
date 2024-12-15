import numpy as np
import sys
import profile
import pstats

IMAX = 100000
ERRO = 1e-6

#--------------------------------------------------------------------------
def bicg(A, x):
    nlin, ncols = A.shape # pega o tamanho da matriz
    
    x = np.zeros((nlin,1)) # x se torna uma lista zerada [0,0,0,0,0]
    r = np.subtract(b,np.matmul(A,x)) # subtrai do vetor B ([1,1,1,1,1] os valores da multiplicação de A por x. Como x é zerado, o resultado é zerado, e b - tudo zerado fica igual a B, que é [1,1,1,1,1]
    r2 = r
    #logo: r, r2 = b

    p = np.zeros((nlin,1)) # zera p e p2
    p2 = np.zeros((nlin,1))
    rho = 1

    i = 1 # contabiliza as iterações

    while  i < IMAX:
        rho0 = rho
        rho = np.dot(np.transpose(r2), r) # usa transpose para transformar mudar orientação do vetor r2 (resulta na mesma coisa) e multiplica todos os valores de r2 pelo indice equivalente em r e depois soma os resultados
        #na primeira iteracao, rho sera 5 
        beta = rho / rho0 # divide o rho da iteracao atual pelo rho da iteracao anterior. Beta comeca com 5
        p = np.add(r,np.multiply(beta,p)) # multiplica a constante beta (5) pelo p da iteracao anterior (comeca zerado), retornando um vetor
        # depois, adiciona cada indice do vetor resultante ao indice equivalente do vetor r ([1,1,1,1,1]). O primeiro resultado é r  
        p2 = np.add(r2,np.multiply(beta,p2)) # mantém o mesmo valor de p, uma vez que r2 e p2 são iguais a r e p
        v = np.matmul(A,p) # multiplica a matriz de origem A pelo valor encontrado em p ([1,1,1,1,1]) e armazena em v
        # o primeiro resultado é [4,5,5,5,10] (a soma de cada linha, multiplicada pela posição equivalente em p)
        alpha = rho/np.dot(np.transpose(p2), v) # alpha vira a divisão de rho pela soma das multiplicações de cada valor em p2 ([1,1,1,1,1]) pelo indice equivalente em v ([4,5,5,5,10])
        # primeiro alpha é 5/29 = 0,17241
        x = np.add(x,np.multiply(alpha,p)) # x, o vetor dos resultados, se torna a soma de cada elemento de x com o equivalente em índice da multiplicação da constante alpha por cada elemento do vetor p ([1,1,1,1,1])
        # na primeira iteração, x é [0,0,0,0,0] + [0.17241,0.17241,0.17241,0.17241,0.17241] = [0.17241,0.17241,0.17241,0.17241,0.17241]
        
        if np.dot(np.transpose(r), r) < ERRO * ERRO: #se soma dos quadrados de cada posição de r([1,1,1,1,1]) for menor que o quadrado estipulado para o erro, para de iterar
            break

        r = np.subtract(r, np.multiply(alpha,v)) # redefine o valor de r para a subtração do valor atual de r ([1,1,1,1,1]) pela multiplicação da constante alpha (0,17241) por cada elemento em v ([4,5,5,5,10])
        # na primeira iteração, r muda para [1,1,1,1,1] - 0,17241*[4,5,5,5,10] = [1,1,1,1,1] - [0.68964,0.86205,0.86205,0.86205,1.7241] = [0.31036,0.13795,0.13795,0.13795,-0.7241]
        r2 = np.subtract(r2, np.multiply(alpha, np.matmul(np.transpose(A),p2))) # redefine o valor de r para a subtração do valor atual de r2 ([1,1,1,1,1]) pela multiplicação da constante alpha (0,17241) pela multiplicação matricial da transposição de A por p2([1,1,1,1,1])
        # na primeira iteração: [1,1,1,1,1] - 0,17241*[10,5,5,5,4] = [-0.7241,0.13795,0.13795,0.13795,0.31036]
        #print(r)
        # print(r2)
        i = i + 1

    return x, i # retorna o vetor resultante (solução para cada variável) e o número de iterações realizadas

n = 5
A = np.array([
    [4, 0, 0, 0, 0],
    [1, 4, 0, 0, 0],
    [0, 1, 4, 0, 0],
    [0, 0, 1, 4, 0],
    [5, 0, 0, 1, 4]
])
b = np.ones((n,1))

x, niter = bicg(A,b) 
print('Iteracoes: %d' % niter)

print(x)
b = np.matmul(A,x)
print(b)