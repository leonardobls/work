#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IMAX 100000
#define ERRO 1e-6 // Ajustado para alinhar ao Python
#define N 5

typedef struct
{
    double x[N];   // Vetor resultado
    int iteracoes; // Número de iterações
} BIGCG_Resultado;

void printVector(int size, double vector[N])
{
    printf("Vetor:\n");
    for (int i = 0; i < size; i++)
    {
        printf("%.16f ", vector[i]);
    }
    printf("\n\n");
}

BIGCG_Resultado bicg(double matrizA[N][N], double vetorB[N])
{
    BIGCG_Resultado retorno;

    // Inicializações
    double x[N] = {0.0};
    double r[N], r2[N], p[N] = {0.0}, p2[N] = {0.0};
    double v[N], temp[N];

    // r = b - A*x (inicialmente x é zero, então r = b)
    for (int i = 0; i < N; i++)
    {
        r[i] = vetorB[i];
        r2[i] = vetorB[i];
    }

    double rho = 1.0;
    int iteracoes = 1;

    // Laço principal
    while (iteracoes < IMAX)
    {
        // Calcula rho
        double rho0 = rho;
        rho = 0.0;
        for (int i = 0; i < N; i++)
        {
            rho += r2[i] * r[i];
        }

        // Calcula beta e atualiza p e p2
        double beta = rho / rho0;
        for (int i = 0; i < N; i++)
        {
            p[i] = r[i] + beta * p[i];
            p2[i] = r2[i] + beta * p2[i];
        }

        // Multiplica matriz A por p para obter v
        for (int i = 0; i < N; i++)
        {
            v[i] = 0.0;
            for (int j = 0; j < N; j++)
            {
                v[i] += matrizA[i][j] * p[j];
            }
        }

        // Calcula denom e alpha
        double denom = 0.0;
        for (int i = 0; i < N; i++)
        {
            denom += p2[i] * v[i];
        }

        double alpha = rho / denom;

        // Atualiza x
        for (int i = 0; i < N; i++)
        {
            x[i] += alpha * p[i];
        }

        // Atualiza r
        double norm_r = 0.0;
        for (int i = 0; i < N; i++)
        {
            r[i] -= alpha * v[i];
            norm_r += r[i] * r[i];
        }

        // Critério de parada
        if (norm_r < ERRO * ERRO)
        {
            break;
        }

        // Atualiza r2
        for (int i = 0; i < N; i++)
        {
            temp[i] = 0.0;
            for (int j = 0; j < N; j++)
            {
                temp[i] += matrizA[j][i] * p2[j];
            }
            r2[i] -= alpha * temp[i];
        }

        iteracoes++;
    }

    for (int i = 0; i < N; i++)
    {
        retorno.x[i] = x[i];
    }
    retorno.iteracoes = iteracoes;

    return retorno;
}

int main()
{
    double A[N][N] = {
        {4, 0, 0, 0, 0},
        {1, 4, 0, 0, 0},
        {0, 1, 4, 0, 0},
        {0, 0, 1, 4, 0},
        {5, 0, 0, 1, 4}};

    double b[N] = {1, 1, 1, 1, 1};

    BIGCG_Resultado resultado = bicg(A, b);

    printf("\nResultado Final:\n");
    printf("Iteracoes: %d\n", resultado.iteracoes);
    printf("x:\n");
    printVector(N, resultado.x);

    double b_calc[N];
    for (int i = 0; i < N; i++)
    {
        b_calc[i] = 0.0;
        for (int j = 0; j < N; j++)
        {
            b_calc[i] += A[i][j] * resultado.x[j];
        }
    }
    printf("b calculado:\n");
    printVector(N, b_calc);

    return 0;
}
