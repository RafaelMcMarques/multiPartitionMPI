#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

long long geraAleatorioLL(int rank) {
    int a = rand();  // Generates a random integer
    int b = rand();  // Another random integer
    long long v = (long long)a * 100 + b;
    return v;
}

// Function to partition the input and distribute it to appropriate processes
void multi_partition_mpi(long long* Input, int n, long long* P, int np, 
                          long long* Output, int* Pos, int rank, int world_size) {

    // Determine the number of elements each process will handle
    int local_n = n / world_size;
    int start_idx = rank * local_n;
    int end_idx = (rank == world_size - 1) ? n : (rank + 1) * local_n; // Handle the last process

    // Partitioner: Assign values to local partitions
    int* local_count = (int*)malloc(np * sizeof(int));
    for (int i = 0; i < np; ++i) {
        local_count[i] = 0;
    }

    // Count elements that belong to each range
    for (int i = start_idx; i < end_idx; ++i) {
        int faixa = -1;
        for (int j = 0; j < np; ++j) {
            if (Input[i] < P[j]) {
                faixa = j;
                break;
            }
        }
        if (faixa != -1) {
            local_count[faixa]++;
        }
    }

    // Gather the global counts of each range (sum over all processes)
    int* global_count = (int*)malloc(np * sizeof(int));
    MPI_Allreduce(local_count, global_count, np, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Calculate positions of each range in the global output
    Pos[0] = 0;
    for (int i = 1; i < np; ++i) {
        Pos[i] = Pos[i - 1] + global_count[i - 1];
    }

    // Now distribute the input data into the appropriate output locations
    int* local_pos = (int*)malloc(np * sizeof(int));
    for (int i = 0; i < np; ++i) {
        local_pos[i] = 0;
    }

    // Redistribute local data
    for (int i = start_idx; i < end_idx; ++i) {
        int faixa = -1;
        for (int j = 0; j < np; ++j) {
            if (Input[i] < P[j]) {
                faixa = j;
                break;
            }
        }
        if (faixa != -1) {
            Output[Pos[faixa] + local_pos[faixa]] = Input[i];
            local_pos[faixa]++;
        }
    }

    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);

    free(local_count);
    free(global_count);
    free(local_pos);
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Generate a random seed based on rank
    int seed = 2024 * 100 + world_rank;
    srand(seed);

    // Define the number of elements in Input
    const int nTotal = 16000000;
    const int np = world_size;  // The number of partitions should be equal to world_size

    // Generate the vector P (same for all processes)
    long long P[np];
    for (int i = 0; i < np; ++i) {
        P[i] = (i + 1) * 100;  // Example range values
    }

    // Generate the Input vector (each process generates nTotal / world_size elements)
    int local_n = nTotal / world_size;
    long long* Input = (long long*)malloc(local_n * sizeof(long long));
    for (int i = 0; i < local_n; ++i) {
        Input[i] = geraAleatorioLL(world_rank);  // Generate random numbers
    }

    // Prepare for the output array and partition positions
    long long* Output = (long long*)malloc(nTotal * sizeof(long long));
    int* Pos = (int*)malloc(np * sizeof(int));

    // Call the partition function to distribute data across processes
    multi_partition_mpi(Input, nTotal, P, np, Output, Pos, world_rank, world_size);

    // Output first 10 values from the global Output (rank 0 does this)
    if (world_rank == 0) {
        printf("First 10 elements in the Output:\n");
        for (int i = 0; i < 10; ++i) {
            printf("%lld ", Output[i]);
        }
        printf("\n");
    }

    // Clean up
    free(Input);
    free(Output);
    free(Pos);

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}
