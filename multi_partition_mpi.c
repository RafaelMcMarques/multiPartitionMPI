#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "chrono.c"

#define NTIMES 10
#define DEBUG 0

int verifica = 0;

// retorna indice do primeiro elemento estritamente MAIOR que x
int upper_bound(long long x, long long *v, int n) {
    int first = 0, last = n - 1, ans = -1; 
    while (first <= last) {
        int m = first + (last - first) / 2;

        if (v[m] > x) {
            ans = m;
            last = m - 1;
        } else {
            first = m + 1;
        }
    }

    return ans;
}


void particiona(long long* Input, int n, long long* P, int np, long long **particoes, int *tamanho_particoes) {
    for (int i = 0; i < n; i++) {
        int particao = upper_bound(Input[i], P, np);
        particoes[particao][tamanho_particoes[particao]++] = Input[i];
    }
}

void verifica_particoes(int rank, int total_recebido, long long *minha_particao, int n, long long* P, int np) {
    int soma_total;
    MPI_Reduce(&total_recebido, &soma_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // O processo 0 exibe a soma total
    if (rank == 0) {
        printf("Elementos particionados: %d\n", soma_total);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    long long lim_inf = rank == 0 ? 0 : P[rank - 1];
    long long lim_sup = P[rank];

    for (int i = 0; i < n; i++) {
        if ((minha_particao[i]) < lim_inf || (minha_particao[i] >= lim_sup)) {
            printf("PARTICAO %d COM ERROS, encontrado %lld\n", rank, minha_particao[i]);
            return;
        }
    }
    printf("PARTICAO %d CORRETA!\n", rank);
}

void multi_partition_mpi(long long *Input, int n, long long *P, int np, int rank) {
    long long **particoes = (long long**) malloc(np * sizeof(long long*));
    for (int i = 0; i < np; ++i) {
        particoes[i] = (long long*) malloc(n * sizeof(long long)); // Alocação máxima possível
    }
    int *tamanho_particoes = (int*) calloc(np, sizeof(int));

    particiona(Input, n, P, np, particoes, tamanho_particoes);

    // Preparar dados para MPI_Alltoallv
    int *sendcounts = (int*)malloc(np * sizeof(int));
    int *sdispls = (int*)malloc(np * sizeof(int));
    int *recvcounts = (int*)malloc(np * sizeof(int));
    int *rdispls = (int*)malloc(np * sizeof(int));

    // Calculando sendcounts e deslocamentos
    int total_envio = 0;
    for (int i = 0; i < np; ++i) {
        sendcounts[i] = tamanho_particoes[i];
        sdispls[i] = total_envio;
        total_envio += tamanho_particoes[i];
    }

    long long *sendbuf = (long long*)malloc(total_envio * sizeof(long long));
    for (int i = 0, idx = 0; i < np; ++i) {
        for (int j = 0; j < tamanho_particoes[i]; ++j) {
            sendbuf[idx++] = particoes[i][j];
        }
    }

    // Receber tamanhos de cada processo
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    // Calculando rdispls
    int total_recebido = 0;
    for (int i = 0; i < np; ++i) {
        rdispls[i] = total_recebido;
        total_recebido += recvcounts[i];
    }

    long long *minha_particao = (long long*)malloc(total_recebido * sizeof(long long));

    // Realizar comunicação com MPI_Alltoallv
    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_LONG_LONG,
                  minha_particao, recvcounts, rdispls, MPI_LONG_LONG,
                  MPI_COMM_WORLD);

    // Conferir resultados
    if (verifica) {
        verifica_particoes(rank, total_recebido, minha_particao, total_recebido, P, np);
    }

    // Libera memória
    for (int i = 0; i < np; ++i) {
        free(particoes[i]);
    }
    free(particoes);
    free(tamanho_particoes);
    free(sendcounts);
    free(sdispls);
    free(recvcounts);
    free(rdispls);
    free(sendbuf);
    free(minha_particao);

    MPI_Barrier(MPI_COMM_WORLD);
}

int compare(const void *a, const void *b) {
    const long long int_a = *(const long long *)a;
    const long long int_b = *(const long long *)b;

    if (int_a < int_b)
        return -1;
    else if (int_a > int_b)
        return 1;
    else
        return 0;
}


long long geraAleatorioLL() {
      int a = rand();  // Returns a pseudo-random integer
	               //    between 0 and RAND_MAX.
      int b = rand();  // same as above
      long long v = (long long)a * 100 + b;
      return v;
} 

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // checar se parametro -v foi passsado
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-v") == 0) {
            verifica = 1;
            break;
        }
    }

    int np, rank, nTotalElements = atoi(argv[1]);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(2024 * 100 + rank);

    // Generate the vector P (same for all processes)
    long long *P = (long long*)malloc(np * sizeof(long long));
    if (rank == 0) {
        for (int i = 0; i < np-1; ++i) {
            P[i] = geraAleatorioLL(); 
        }
        P[np-1] = LLONG_MAX;
        qsort(P, np, sizeof(long long), compare);
    }
    MPI_Bcast(P, np, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Generate the Input vector (each process generates nTotal / world_size elements)
    int local_n = nTotalElements / np; 
    if ((nTotalElements%np != 0) && (rank == np - 1)) 
        local_n += nTotalElements%np;

    long long* Input = (long long*)malloc(local_n * sizeof(long long));
    for (int i = 0; i < local_n; ++i) {
        Input[i] = geraAleatorioLL();
    }

    if (DEBUG && rank == 0) {
        printf("\nP: ");
        for (int i = 0; i < np; i++) printf("%lld ", P[i]);
        printf("\n\n");
    }

    chronometer_t chrono_time;
    chrono_reset(&chrono_time);
    chrono_start(&chrono_time);

    for (int i = 0; i < NTIMES; i++) {
        if (rank == 0) {
            printf("\n==== CALL %d ====\n\n", i);
        }
        multi_partition_mpi(Input, local_n, P, np, rank);
    } 

    chrono_stop(&chrono_time);

    if (rank == 0) {
        double total_time_in_seconds = (double)chrono_gettotal(&chrono_time) / ((double)1000 * 1000 * 1000);
        printf("\nTotal Time: %lfs\n", total_time_in_seconds);
        printf("Throughput: %.2lfMEPS/s\n", nTotalElements / (total_time_in_seconds * 1e6));
    }


    free(P);
    free(Input);

    // Finalize the MPI environment.
    MPI_Finalize();
}

