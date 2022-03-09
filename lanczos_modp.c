/*
 * Sequential implementation of the Block-Lanczos algorithm.
 *
 * This is based on the paper:
 *     "A modified block Lanczos algorithm with fewer vectors"
 *
 *  by Emmanuel Thomé, available online at
 *      https://hal.inria.fr/hal-01293351/document
 *
 * Authors : Charles Bouillaguet
 *
 * v1.00 (2022-01-18)
 *
 * USAGE:
 *      $ ./lanczos_modp --prime 65537 --n 4 --matrix random_small.mtx
 *
 */
#define _POSIX_C_SOURCE 1 // ctime
#include <inttypes.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <err.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <assert.h>
#include <mpi.h>
#include <mmio.h>

typedef uint64_t u64;
typedef uint32_t u32;

/******************* global variables ********************/

long n = 1;
u64 prime;
char *matrix_filename;
char *kernel_filename;
bool right_kernel = false;
int stop_after = -1;

int n_iterations; /* variables of the "verbosity engine" */
double start;
double last_print;
bool ETA_flag;
int expected_iterations;

/******************* sparse matrix data structure **************/

struct sparsematrix_t
{
        int nrows; // dimensions
        int ncols;
        long int nnz; // number of non-zero coefficients
        int *i;       // row indices (for COO matrices)
        int *j;       // column indices
        u32 *x;       // coefficients
};

/******************* pseudo-random generator (xoshiro256+) ********************/

/* fixed seed --- this is bad */
u64 rng_state[4] = {0x1415926535, 0x8979323846, 0x2643383279, 0x5028841971};

u64 rotl(u64 x, int k)
{
        u64 foo = x << k;
        u64 bar = x >> (64 - k);
        return foo ^ bar;
}

u64 random64()
{
        u64 result = rotl(rng_state[0] + rng_state[3], 23) + rng_state[0];
        u64 t = rng_state[1] << 17;
        rng_state[2] ^= rng_state[0];
        rng_state[3] ^= rng_state[1];
        rng_state[1] ^= rng_state[2];
        rng_state[0] ^= rng_state[3];
        rng_state[2] ^= t;
        rng_state[3] = rotl(rng_state[3], 45);
        return result;
}

/******************* utility functions ********************/

double wtime()
{
        struct timeval ts;
        gettimeofday(&ts, NULL);
        return (double)ts.tv_sec + ts.tv_usec / 1e6;
}

/* represent n in <= 6 char  */
void human_format(char *target, long n)
{
        if (n < 1000)
        {
                sprintf(target, "%" PRId64, n);
                return;
        }
        if (n < 1000000)
        {
                sprintf(target, "%.1fK", n / 1e3);
                return;
        }
        if (n < 1000000000)
        {
                sprintf(target, "%.1fM", n / 1e6);
                return;
        }
        if (n < 1000000000000ll)
        {
                sprintf(target, "%.1fG", n / 1e9);
                return;
        }
        if (n < 1000000000000000ll)
        {
                sprintf(target, "%.1fT", n / 1e12);
                return;
        }
}

/************************** command-line options ****************************/

void usage(char **argv)
{
        printf("%s [OPTIONS]\n\n", argv[0]);
        printf("Options:\n");
        printf("--matrix FILENAME           MatrixMarket file containing the spasre matrix\n");
        printf("--prime P                   compute modulo P\n");
        printf("--n N                       blocking factor [default 1]\n");
        printf("--output-file FILENAME      store the block of kernel vectors\n");
        printf("--right                     compute right kernel vectors\n");
        printf("--left                      compute left kernel vectors [default]\n");
        printf("--stop-after N              stop the algorithm after N iterations\n");
        printf("\n");
        printf("The --matrix and --prime arguments are required\n");
        printf("The --stop-after and --output-file arguments mutually exclusive\n");
        exit(0);
}

void process_command_line_options(int argc, char **argv)
{
        struct option longopts[8] = {
            {"matrix", required_argument, NULL, 'm'},
            {"prime", required_argument, NULL, 'p'},
            {"n", required_argument, NULL, 'n'},
            {"output-file", required_argument, NULL, 'o'},
            {"right", no_argument, NULL, 'r'},
            {"left", no_argument, NULL, 'l'},
            {"stop-after", required_argument, NULL, 's'},
            {NULL, 0, NULL, 0}};
        char ch;
        while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1)
        {
                switch (ch)
                {
                case 'm':
                        matrix_filename = optarg;
                        break;
                case 'n':
                        n = atoi(optarg);
                        break;
                case 'p':
                        prime = atoll(optarg);
                        break;
                case 'o':
                        kernel_filename = optarg;
                        break;
                case 'r':
                        right_kernel = true;
                        break;
                case 'l':
                        right_kernel = false;
                        break;
                case 's':
                        stop_after = atoll(optarg);
                        break;
                default:
                        errx(1, "Unknown option\n");
                }
        }

        /* missing required args? */
        if (matrix_filename == NULL || prime == 0)
                usage(argv);
        /* exclusive arguments? */
        if (kernel_filename != NULL && stop_after > 0)
                usage(argv);
        /* range checking */
        if (prime > 0x3fffffdd)
        {
                errx(1, "p is capped at 2**30 - 35.  Slighly larger values could work, with the\n");
                printf("suitable code modifications.\n");
                exit(1);
        }
}

/****************** sparse matrix operations ******************/

/* Load a matrix from a file in "list of triplet" representation */
void sparsematrix_mm_load(struct sparsematrix_t *M_processus, char const *filename, int my_rank, int p)
{
        int MSG_TAG = 10;

        if (my_rank == 0)
        {
                int nrows = 0;
                int ncols = 0;
                long nnz = 0;

                printf("Loading matrix from %s\n", filename);
                fflush(stdout);

                FILE *f = fopen(filename, "r");

                if (f == NULL)
                        err(1, "impossible d'ouvrir %s", filename);

                /* read the header, check format */
                MM_typecode matcode;
                if (mm_read_banner(f, &matcode) != 0)
                        errx(1, "Could not process Matrix Market banner.\n");
                if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode))
                        errx(1, "Matrix Market type: [%s] not supported (only sparse matrices are OK)",
                             mm_typecode_to_str(matcode));
                if (!mm_is_general(matcode) || !mm_is_integer(matcode))
                        errx(1, "Matrix type [%s] not supported (only integer general are OK)",
                             mm_typecode_to_str(matcode));
                if (mm_read_mtx_crd_size(f, &nrows, &ncols, &nnz) != 0)
                        errx(1, "Cannot read matrix size");

                fprintf(stderr, "  - [%s] %d x %d with %ld nz\n", mm_typecode_to_str(matcode), nrows, ncols, nnz);
                fprintf(stderr, "  - Allocating %.1f MByte\n", 1e-6 * (12.0 * nnz));

                /* Allocate memory for the matrix */

                // liste des indices i
                int *Mi = malloc(nnz * sizeof(*Mi));
                // liste des indices j
                int *Mj = malloc(nnz * sizeof(*Mj));

                // liste des valeurs M[i,j]
                u32 *Mx = malloc(nnz * sizeof(*Mx));

                if (Mi == NULL || Mj == NULL || Mx == NULL)
                        err(1, "Cannot allocate sparse matrix");

                /* Parse and load actual entries */
                double start = wtime();

                for (long u = 0; u < nnz; u++)
                {
                        int i, j;
                        u32 x;
                        if (3 != fscanf(f, "%d %d %d\n", &i, &j, &x))
                                errx(1, "parse error entry %ld\n", u);
                        Mi[u] = i - 1; /* MatrixMarket is 1-based */
                        Mj[u] = j - 1;
                        Mx[u] = x % prime;

                        // verbosity
                        if ((u & 0xffff) == 0xffff)
                        {
                                double elapsed = wtime() - start;
                                double percent = (100. * u) / nnz;
                                double rate = ftell(f) / 1048576. / elapsed;
                                printf("\r  - Reading %s: %.1f%% (%.1f MB/s)", matrix_filename, percent, rate);
                        }
                }

                /* finalization */
                fclose(f);
                printf("\n");

                /***** PARTIE TRANSFERT DE DONNEES *****/

                int bloc_size = (nnz / p);

                // je keep les infos du processus 0s

                M_processus->nrows = nrows;
                M_processus->ncols = ncols;
                M_processus->nnz = bloc_size;
                M_processus->i = malloc(bloc_size * sizeof(*M_processus->i));
                M_processus->j = malloc(bloc_size * sizeof(*M_processus->j));
                M_processus->x = malloc(bloc_size * sizeof(*M_processus->x));

                for (int u = 0; u < bloc_size; u++)
                {
                        M_processus->i[u] = Mi[u];
                        M_processus->j[u] = Mj[u];
                        M_processus->x[u] = Mx[u];
                }

                long int step, new_nnz;

                // j'envoie le reste aux autres processus
                for (int i = 1; i < p; i++)
                {
                        step = 0;
                        // les lignes restantes sont données au dernier processus
                        if (i == p - 1)
                        {
                                step = (nnz % p);
                        }

                        new_nnz = bloc_size + step;

                        MPI_Send(&nrows, 1, MPI_INT, i, MSG_TAG, MPI_COMM_WORLD);
                        MPI_Send(&ncols, 1, MPI_INT, i, MSG_TAG, MPI_COMM_WORLD);
                        MPI_Send(&new_nnz, 1, MPI_LONG_INT, i, MSG_TAG, MPI_COMM_WORLD);

                        MPI_Send(&Mi[i * bloc_size], new_nnz, MPI_INT, i, MSG_TAG, MPI_COMM_WORLD);
                        MPI_Send(&Mj[i * bloc_size], new_nnz, MPI_INT, i, MSG_TAG, MPI_COMM_WORLD);
                        MPI_Send(&Mx[i * bloc_size], new_nnz, MPI_UINT32_T, i, MSG_TAG, MPI_COMM_WORLD);
                }
        }

        else
        {

                // les autres processus receptionnent leurs vals
                MPI_Recv(&(M_processus->nrows), 1, MPI_INT, 0, MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&(M_processus->ncols), 1, MPI_INT, 0, MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&(M_processus->nnz), 1, MPI_LONG_INT, 0, MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // has to be allocated first
                M_processus->i = malloc((M_processus->nnz) * sizeof(*M_processus->i));
                M_processus->j = malloc((M_processus->nnz) * sizeof(*M_processus->j));
                M_processus->x = malloc((M_processus->nnz) * sizeof(*M_processus->x));

                MPI_Recv(M_processus->i, M_processus->nnz, MPI_INT, 0, MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(M_processus->j, M_processus->nnz, MPI_INT, 0, MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(M_processus->x, M_processus->nnz, MPI_UINT32_T, 0, MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
}

/* y += M*x or y += transpose(M)*x, according to the transpose flag */
void sparse_matrix_vector_product(u32 *y, struct sparsematrix_t const *M, u32 const *x, bool transpose)
{
        long nnz = M->nnz;
        int nrows = transpose ? M->ncols : M->nrows;
        int const *Mi = M->i;
        int const *Mj = M->j;
        u32 const *Mx = M->x;

        for (long i = 0; i < nrows * n; i++)
                y[i] = 0;

        for (long k = 0; k < nnz; k++)
        {
                int i = transpose ? Mj[k] : Mi[k];
                int j = transpose ? Mi[k] : Mj[k];
                u64 v = Mx[k];
                for (int l = 0; l < n; l++)
                {
                        u64 a = y[i * n + l];
                        u64 b = x[j * n + l];
                        y[i * n + l] = (a + v * b) % prime;
                }
        }
}

/****************** dense linear algebra modulo p *************************/

/* C += A*B   for n x n matrices */
void matmul_CpAB(u32 *C, u32 const *A, u32 const *B)
{
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        for (int k = 0; k < n; k++)
                        {
                                u64 x = C[i * n + j];
                                u64 y = A[i * n + k];
                                u64 z = B[k * n + j];
                                C[i * n + j] = (x + y * z) % prime;
                        }
}

/* C += transpose(A)*B   for n x n matrices */
void matmul_CpAtB(u32 *C, u32 const *A, u32 const *B)
{
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        for (int k = 0; k < n; k++)
                        {
                                u64 x = C[i * n + j];
                                u64 y = A[k * n + i];
                                u64 z = B[k * n + j];
                                C[i * n + j] = (x + y * z) % prime;
                        }
}

/* return a^(-1) mod b */
u32 invmod(u32 a, u32 b)
{
        long int t = 0;
        long int nt = 1;
        long int r = b;
        long int nr = a % b;
        while (nr != 0)
        {
                long int q = r / nr;
                long int tmp = nt;
                nt = t - q * nt;
                t = tmp;
                tmp = nr;
                nr = r - q * nr;
                r = tmp;
        }
        if (t < 0)
                t += b;
        return (u32)t;
}

/*
 * Given an n x n matrix U, compute a "partial-inverse" V and a diagonal matrix
 * d such that d*V == V*d == V and d == V*U*d. Return the number of pivots.
 */
int semi_inverse(u32 const *M_, u32 *winv, u32 *d)
{
        u32 M[n * n];
        int npiv = 0;
        for (int i = 0; i < n * n; i++) /* copy M <--- M_ */
                M[i] = M_[i];
        /* phase 1: compute d */
        for (int i = 0; i < n; i++) /* setup d */
                d[i] = 0;
        for (int j = 0; j < n; j++)
        { /* search a pivot on column j */
                int pivot = n;
                for (int i = j; i < n; i++)
                        if (M[i * n + j] != 0)
                        {
                                pivot = i;
                                break;
                        }
                if (pivot >= n)
                        continue; /* no pivot found */
                d[j] = 1;
                npiv += 1;
                u64 pinv = invmod(M[pivot * n + j], prime); /* multiply pivot row to make pivot == 1 */
                for (int k = 0; k < n; k++)
                {
                        u64 x = M[pivot * n + k];
                        M[pivot * n + k] = (x * pinv) % prime;
                }
                for (int k = 0; k < n; k++)
                { /* swap pivot row with row j */
                        u32 tmp = M[j * n + k];
                        M[j * n + k] = M[pivot * n + k];
                        M[pivot * n + k] = tmp;
                }
                for (int i = 0; i < n; i++)
                { /* eliminate everything else on column j */
                        if (i == j)
                                continue;
                        u64 multiplier = M[i * n + j];
                        for (int k = 0; k < n; k++)
                        {
                                u64 x = M[i * n + k];
                                u64 y = M[j * n + k];
                                M[i * n + k] = (x + (prime - multiplier) * y) % prime;
                        }
                }
        }
        /* phase 2: compute d and winv */
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                        M[i * n + j] = (d[i] && d[j]) ? M_[i * n + j] : 0;
                        winv[i * n + j] = ((i == j) && d[i]) ? 1 : 0;
                }
        npiv = 0;
        for (int i = 0; i < n; i++)
                d[i] = 0;
        /* same dance */
        for (int j = 0; j < n; j++)
        {
                int pivot = n;
                for (int i = j; i < n; i++)
                        if (M[i * n + j] != 0)
                        {
                                pivot = i;
                                break;
                        }
                if (pivot >= n)
                        continue;
                d[j] = 1;
                npiv += 1;
                u64 pinv = invmod(M[pivot * n + j], prime);
                for (int k = 0; k < n; k++)
                {
                        u64 x = M[pivot * n + k];
                        M[pivot * n + k] = (x * pinv) % prime;
                }
                for (int k = 0; k < n; k++)
                {
                        u32 tmp = M[j * n + k];
                        M[j * n + k] = M[pivot * n + k];
                        M[pivot * n + k] = tmp;
                }
                for (int k = 0; k < n; k++)
                {
                        u64 x = winv[pivot * n + k];
                        winv[pivot * n + k] = (x * pinv) % prime;
                }
                for (int k = 0; k < n; k++)
                {
                        u32 tmp = winv[j * n + k];
                        winv[j * n + k] = winv[pivot * n + k];
                        winv[pivot * n + k] = tmp;
                }
                for (int i = 0; i < n; i++)
                {
                        if (i == j)
                                continue;
                        u64 multiplier = M[i * n + j];
                        for (int k = 0; k < n; k++)
                        {
                                u64 x = M[i * n + k];
                                u64 y = M[j * n + k];
                                M[i * n + k] = (x + (prime - multiplier) * y) % prime;
                                u64 w = winv[i * n + k];
                                u64 z = winv[j * n + k];
                                winv[i * n + k] = (w + (prime - multiplier) * z) % prime;
                        }
                }
        }
        return npiv;
}

/*************************** block-Lanczos algorithm ************************/

/* Computes vtAv <-- transpose(v) * Av, vtAAv <-- transpose(v) * Av */
void block_dot_products(u32 *vtAv, u32 *vtAAv, int N, u32 const *Av, u32 const *v)
{
        for (int i = 0; i < n * n; i++)
                vtAv[i] = 0;
        for (int i = 0; i < N; i += n)
                matmul_CpAtB(vtAv, &v[i * n], &Av[i * n]);

        for (int i = 0; i < n * n; i++)
                vtAAv[i] = 0;
        for (int i = 0; i < N; i += n)
                matmul_CpAtB(vtAAv, &Av[i * n], &Av[i * n]);
}

/* Compute the next values of v (in tmp) and p */
void orthogonalize(u32 *v, u32 *tmp, u32 *p, u32 *d, u32 const *vtAv, const u32 *vtAAv,
                   u32 const *winv, int N, u32 const *Av)
{
        /* compute the n x n matrix c */
        u32 c[n * n];
        u32 spliced[n * n];
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                        spliced[i * n + j] = d[j] ? vtAAv[i * n + j] : vtAv[i * n + j];
                        c[i * n + j] = 0;
                }
        matmul_CpAB(c, winv, spliced);
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        c[i * n + j] = prime - c[i * n + j];

        u32 vtAvd[n * n];
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        vtAvd[i * n + j] = d[j] ? prime - vtAv[i * n + j] : 0;

        /* compute the next value of v ; store it in tmp */
        for (long i = 0; i < N; i++)
                for (long j = 0; j < n; j++)
                        tmp[i * n + j] = d[j] ? Av[i * n + j] : v[i * n + j];
        for (long i = 0; i < N; i += n)
                matmul_CpAB(&tmp[i * n], &v[i * n], c);
        for (long i = 0; i < N; i += n)
                matmul_CpAB(&tmp[i * n], &p[i * n], vtAvd);

        /* compute the next value of p */
        for (long i = 0; i < N; i++)
                for (long j = 0; j < n; j++)
                        p[i * n + j] = d[j] ? 0 : p[i * n + j];
        for (long i = 0; i < N; i += n)
                matmul_CpAB(&p[i * n], &v[i * n], winv);
}

void verbosity()
{
        n_iterations += 1;
        double elapsed = wtime() - start;
        if (elapsed - last_print < 1)
                return;

        last_print = elapsed;
        double per_iteration = elapsed / n_iterations;
        double estimated_length = expected_iterations * per_iteration;
        time_t end = start + estimated_length;
        if (!ETA_flag)
        {
                int d = estimated_length / 86400;
                estimated_length -= d * 86400;
                int h = estimated_length / 3600;
                estimated_length -= h * 3600;
                int m = estimated_length / 60;
                estimated_length -= m * 60;
                int s = estimated_length;
                printf("    - Expected duration : ");
                if (d > 0)
                        printf("%d j ", d);
                if (h > 0)
                        printf("%d h ", h);
                if (m > 0)
                        printf("%d min ", m);
                printf("%d s\n", s);
                ETA_flag = true;
        }
        char ETA[30];
        ctime_r(&end, ETA);
        ETA[strlen(ETA) - 1] = 0; // élimine le \n final
        printf("\r    - iteration %d / %d. %.3fs per iteration. ETA: %s",
               n_iterations, expected_iterations, per_iteration, ETA);
        fflush(stdout);
}

/* optional tests */
void correctness_tests(u32 const *vtAv, u32 const *vtAAv, u32 const *winv, u32 const *d)
{
        /* vtAv, vtAAv, winv are actually symmetric + winv and d match */
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                        assert(vtAv[i * n + j] == vtAv[j * n + i]);
                        assert(vtAAv[i * n + j] == vtAAv[j * n + i]);
                        assert(winv[i * n + j] == winv[j * n + i]);
                        assert((winv[i * n + j] == 0) || d[i] || d[j]);
                }
        /* winv satisfies d == winv * vtAv*d */
        u32 vtAvd[n * n];
        u32 check[n * n];
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                {
                        vtAvd[i * n + j] = d[j] ? vtAv[i * n + j] : 0;
                        check[i * n + j] = 0;
                }
        matmul_CpAB(check, winv, vtAvd);
        for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                        if (i == j)
                                assert(check[j * n + j] == d[i]);
                        else
                                assert(check[i * n + j] == 0);
}

/* check that we actually computed a kernel vector */
void final_check(int nrows, int ncols, u32 const *v, u32 const *vtM)
{
        printf("Final check:\n");
        /* Check if v != 0 */
        bool good = false;
        for (long i = 0; i < nrows; i++)
                for (long j = 0; j < n; j++)
                        good |= (v[i * n + j] != 0);
        if (good)
                printf("  - OK:    v != 0\n");
        else
                printf("  - KO:    v == 0\n");

        /* tmp == Mt * v. Check if tmp == 0 */
        good = true;
        for (long i = 0; i < ncols; i++)
                for (long j = 0; j < n; j++)
                        good &= (vtM[i * n + j] == 0);
        if (good)
                printf("  - OK: vt*M == 0\n");
        else
                printf("  - KO: vt*M != 0\n");
}

/* Solve x*M == 0 or M*x == 0 (if transpose == True) */
u32 *block_lanczos(struct sparsematrix_t const *M, int n, bool transpose)
{
        printf("Block Lanczos\n");

        /************* preparations **************/

        /* allocate blocks of vectors */

        // il check si on a une transposée et inverse les indices si oui
        int nrows = transpose ? M->ncols : M->nrows;
        int ncols = transpose ? M->nrows : M->ncols;

        long block_size = nrows * n;
        // division entiere
        long Npad = ((nrows + n - 1) / n) * n;

        long block_size_pad = Npad * n;

        char human_size[8];

        human_format(human_size, 4 * sizeof(int) * block_size_pad);

        printf("  - Extra storage needed: %sB\n", human_size);

        u32 *v = malloc(sizeof(*v) * block_size_pad);

        u32 *tmp = malloc(sizeof(*tmp) * block_size_pad);

        u32 *Av = malloc(sizeof(*Av) * block_size_pad);

        u32 *p = malloc(sizeof(*p) * block_size_pad);

        if (v == NULL || tmp == NULL || Av == NULL || p == NULL)

                errx(1, "impossible d'allouer les blocs de vecteur");

        /* warn the user */
        expected_iterations = 1 + ncols / n;

        char human_its[8];
        human_format(human_its, expected_iterations);
        printf("  - Expecting %s iterations\n", human_its);

        // more suited initialisation ?

        /* prepare initial values */
        for (long i = 0; i < block_size_pad; i++)
        {
                Av[i] = 0;
                v[i] = 0;
                p[i] = 0;
                tmp[i] = 0;
        }

        for (long i = 0; i < block_size; i++)
                v[i] = random64() % prime;

        /************* main loop *************/
        printf("  - Main loop\n");

        start = wtime();
        bool stop = false;
        while (true)
        {
                if (stop_after > 0 && n_iterations == stop_after)
                        break;

                // tmp = M * v ( equivalent en tTD de y = M x)
                sparse_matrix_vector_product(tmp, M, v, !transpose);
                // Av = M *tmp  (equivalent en td de z = Mtransposé *y)
                sparse_matrix_vector_product(Av, M, tmp, transpose);

                u32 vtAv[n * n];  // xt * z
                u32 vtAAv[n * n]; // zt * z

                // bloc bloc B  = vtAv A = vtAAv
                block_dot_products(vtAv, vtAAv, nrows, Av, v);

                u32 winv[n * n];
                u32 d[n];
                stop = (semi_inverse(vtAv, winv, d) == 0);

                /* check that everything is working ; disable in production */
                correctness_tests(vtAv, vtAAv, winv, d);

                if (stop)
                        break;

                orthogonalize(v, tmp, p, d, vtAv, vtAAv, winv, nrows, Av);

                /* the next value of v is in tmp ; copy */
                for (long i = 0; i < block_size; i++)
                        v[i] = tmp[i];

                verbosity();
        }
        printf("\n");

        if (stop_after < 0)
                final_check(nrows, ncols, v, tmp);
        printf("  - Terminated in %.1fs after %d iterations\n", wtime() - start, n_iterations);
        free(tmp);
        free(Av);
        free(p);
        return v; // x
}

/**************************** dense vector block IO ************************/

void save_vector_block(char const *filename, int nrows, int ncols, u32 const *v)
{
        printf("Saving result in %s\n", filename);
        FILE *f = fopen(filename, "w");
        if (f == NULL)
                err(1, "cannot open %s", filename);
        fprintf(f, "%%%%MatrixMarket matrix array integer general\n");
        fprintf(f, "%%block of left-kernel vector computed by lanczos_modp\n");
        fprintf(f, "%d %d\n", nrows, ncols);
        for (long j = 0; j < ncols; j++)
                for (long i = 0; i < nrows; i++)
                        fprintf(f, "%d\n", v[i * n + j]);
        fclose(f);
}

void getSubMatrix(struct sparsematrix_t M, struct sparsematrix_t *M_processus, int my_rank, int bloc_size)
{

        // je reecup le bloc de p0
        // m not sure if i should give it the same val
        M_processus->ncols = M.ncols;
        M_processus->nrows = M.nrows;
        M_processus->nnz = bloc_size;

        M_processus->x = malloc(bloc_size * sizeof(*M_processus->x));
        M_processus->i = malloc(bloc_size * sizeof(*M_processus->i));
        M_processus->j = malloc(bloc_size * sizeof(*M_processus->j));

        for (int i = my_rank * bloc_size; i < (my_rank + 1) * bloc_size; i++)
        {
                M_processus->x[i] = M.x[i];
                M_processus->i[i] = M.i[i];
                M_processus->j[i] = M.j[i];
        }
}

/*************************** main function *********************************/

int main(int argc, char **argv)
{

        /* init parallelisation */

        int my_rank; /* rang du processeur actuel */
        int p;       /* nombre de processeurs */

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &p);

        // command line options -> nothing to change
        process_command_line_options(argc, argv);

        // loading the matrix
        struct sparsematrix_t M_processus;

        // processus 0 divise M en blocs qu'il donne a chaque processus
        sparsematrix_mm_load(&M_processus, matrix_filename, my_rank, p);

        // // coeur du travail
        // u32 *kernel = block_lanczos(&M, n, right_kernel);

        // if (kernel_filename)
        //         save_vector_block(kernel_filename, right_kernel ? M.ncols : M.nrows, n, kernel);
        // else
        //         printf("Not saving result (no --output given)\n");
        // free(kernel);

        MPI_Finalize();
        exit(EXIT_SUCCESS);
}