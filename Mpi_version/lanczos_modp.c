/*
 * MPI version of the Block-Lanczos algorithm.
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
void sparsematrix_mm_load(struct sparsematrix_t *M, char const *filename)
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

        M->nrows = nrows;
        M->ncols = ncols;
        M->nnz = nnz;
        M->i = Mi;
        M->j = Mj;
        M->x = Mx;
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
        {
                y[i] = 0;
        }

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

/* Computes vtAv <-- transpose(v) * Av, vtAAv <-- transpose(Av) * Av */
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

/*************************** Our functions *********************************/

void transposeMatrix(struct sparsematrix_t *M)
{
        // si on applique pas la transpsée

        int cols = M->ncols;

        M->ncols = M->nrows;

        M->nrows = cols;

        int *i = M->i;

        M->i = M->j;

        M->j = i;
}

// subdivise M par colonnes
struct sparsematrix_t subdiviseM(struct sparsematrix_t M, int nb_processus, int my_rank)
{

        struct sparsematrix_t M_processus;

        if (my_rank == 0)
        {

                /***** PARTIE TRANSFERT DE DONNEES *****/

                int bloc_size = (M.ncols / nb_processus);

                int step;

                // je keep les infos du processus 0

                M_processus.nrows = M.nrows;

                // nb nnz par processus
                long int *nnz_processus = malloc(nb_processus * sizeof(*nnz_processus));
                // nb colonnes par processus
                int *cols_processus = malloc(nb_processus * sizeof(*cols_processus));

                int displs[nb_processus];

                int nnz[nb_processus];

                // on reecupere le nombre de valeurs non nulles par colonnes
                for (int i = 0; i < nb_processus; i++)
                {
                        // les lignes restantes sont données au dernier processus
                        step = (i == nb_processus - 1) ? ((M.ncols) % nb_processus) : 0;

                        nnz_processus[i] = 0;

                        cols_processus[i] = bloc_size + step;
                }

                // computes the number of non null values per cols block
                for (int u = 0; u < M.nnz; u++)
                {
                        for (int i = 0; i < nb_processus; i++)
                        {
                                if (M.j[u] < (i + 1) * cols_processus[i])
                                {
                                        nnz_processus[i]++;
                                        break;
                                }
                        }
                }

                for (int i = 0; i < nb_processus; i++)
                {
                        nnz[i] = nnz_processus[i];
                        fprintf(stderr, "nb of nnz for processus %d = %ld\n", i, nnz_processus[i]);
                }

                M_processus.i = malloc(nnz_processus[0] * sizeof(*M_processus.i));
                M_processus.j = malloc(nnz_processus[0] * sizeof(*M_processus.j));
                M_processus.x = malloc(nnz_processus[0] * sizeof(*M_processus.x));

                // je cumule les nnz de chaque processus pour savoir ou les vals nnz commencent
                long int cum_nnz_processus = nnz_processus[0];

                // // position du 1er element a donner au proc 0
                displs[0] = 0;

                // j'envoie le reste aux autres processus
                for (int i = 1; i < nb_processus; i++)
                {

                        displs[i] = cum_nnz_processus;
                        cum_nnz_processus += nnz_processus[i];
                }

                // je share les infos
                MPI_Bcast(&(M_processus.nrows), 1, MPI_INT, 0, MPI_COMM_WORLD);

                MPI_Scatter(cols_processus, 1, MPI_INT, &(M_processus.ncols), 1, MPI_INT, 0, MPI_COMM_WORLD);

                MPI_Scatter(nnz_processus, 1, MPI_LONG, &(M_processus.nnz), 1, MPI_LONG, 0, MPI_COMM_WORLD);

                MPI_Scatterv(M.i, nnz, displs, MPI_INT, M_processus.i, nnz_processus[0], MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Scatterv(M.j, nnz, displs, MPI_INT, M_processus.j, nnz_processus[0], MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Scatterv(M.x, nnz, displs, MPI_UINT32_T, M_processus.x, nnz_processus[0], MPI_UINT32_T, 0, MPI_COMM_WORLD);
                // fprintf(stderr, "nrows = %d, ncols = %d\n", M_processus.nrows, M_processus.ncols);
        }
        else
        {
                MPI_Bcast(&(M_processus.nrows), 1, MPI_INT, 0, MPI_COMM_WORLD);

                MPI_Scatter(NULL, 0, NULL, &(M_processus.ncols), 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Scatter(NULL, 0, NULL, &(M_processus.nnz), 1, MPI_LONG, 0, MPI_COMM_WORLD);

                // a initialiser
                M_processus.i = malloc((M_processus.nnz) * sizeof(*M_processus.i));
                M_processus.j = malloc((M_processus.nnz) * sizeof(*M_processus.j));
                M_processus.x = malloc((M_processus.nnz) * sizeof(*M_processus.x));

                MPI_Scatterv(NULL, NULL, NULL, NULL, M_processus.i, M_processus.nnz, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Scatterv(NULL, NULL, NULL, NULL, M_processus.j, M_processus.nnz, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Scatterv(NULL, NULL, NULL, NULL, M_processus.x, M_processus.nnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);

                // fprintf(stderr, "nrows = %d, ncols = %d\n", M_processus.nrows, M_processus.ncols);
        }

        // if u wanna verify it

        // if (my_rank == 0)
        // {

        //         FILE *f = fopen("check.mtx", "a+");

        //         for (long u = 0; u < M_processus.nnz; u++)
        //         {
        //                 fprintf(f, "%d \n", M_processus.j[u]);
        //         }

        //         fclose(f);
        // }

        return M_processus;
}

// cree le vecteur V
u32 *createV(int ncols, int nrows)
{
        // fprintf(stderr, "nrows = %d", nrows);
        // fprintf(stderr, "ncols= %d", ncols);

        long block_size = ncols * n;
        // division entiere
        long pad = ((ncols + n - 1) / n) * n;

        long Mpad = ((nrows + n - 1) / n) * n;

        long block_size_pad = (pad > Mpad ? pad : Mpad) * n;

        u32 *v = malloc(sizeof(*v) * block_size_pad);

        // v a la meme valeur pour chaque processus ! no need for transfering or anything
        for (long i = 0; i < block_size; i++)
                v[i] = random64() % prime;

        // better than a double init imo
        for (long i = block_size; i < block_size_pad; i++)
        {
                v[i] = 0;
        }

        return v;
}

// subdivise le vecteur v
u32 *subdiviseV(int ncols, int nrows, int n, int nb_processus, int my_rank)
{
        // the bloc size should

        int ncols_processus = ncols / nb_processus;

        u32 *v_processus;

        int step = 0;

        if (my_rank == 0)
        {

                // partie creation
                u32 *v = createV(ncols, nrows);

                int size = (n * ncols_processus);

                // le processus 0 garde son vecteur v

                v_processus = malloc(sizeof(*v_processus) * size);

                int sendCounts[nb_processus];

                int displs[nb_processus];

                /* partie subdivision */

                for (int proc = 0; proc < nb_processus; proc++)
                {
                        step = (proc == nb_processus - 1) ? (ncols % nb_processus) : 0;
                        sendCounts[proc] = (n * (ncols_processus + step));
                        displs[proc] = proc * size;
                }

                // on subdivise les données
                MPI_Scatterv(v, sendCounts, displs, MPI_UINT32_T, v_processus, n * ncols_processus, MPI_UINT32_T, 0, MPI_COMM_WORLD);

                free(v);
        }

        else
        {

                step = (my_rank == nb_processus - 1) ? (ncols % nb_processus) : 0;

                int receive_size = (n * (ncols_processus + step));

                v_processus = malloc(sizeof(*v_processus) * receive_size);

                MPI_Scatterv(NULL, NULL, NULL, MPI_UINT32_T, v_processus, receive_size, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        }
        // if (my_rank == nb_processus - 1)
        // {
        //         FILE *f = fopen("v0_check.mtx", "a+");

        //         for (int u = 0; u < receive_size; u++)
        //         {
        //                 fprintf(f, "%d \n", v_processus[u]);
        //         }

        //         fclose(f);
        // }

        return v_processus;
}

int getCols(struct sparsematrix_t const M, int my_rank)
{

        int nrows = 0;

        if (my_rank == 0)
        {

                nrows = M.ncols;
        }

        MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);

        return nrows;
}

// operator for the MPI_reduce
void sumMod(void *inputBuffer, void *outputBuffer, int *len, MPI_Datatype *datatype)
{
        // MUTE THAT PARAMETER è.é
        (void)datatype;
        u32 *input = (u32 *)inputBuffer;
        u32 *output = (u32 *)outputBuffer;

        u64 out;

        for (int i = 0; i < *len; i++)
        {

                out = ((u64)output[i] + (u64)input[i]) % prime;
                // output[i] = (output[i] + input[i]) % prime;

                output[i] = (u32)out;
        }
}

// parallel M * x
void computeMatrixVectorProduct(u32 *tmp_processus, struct sparsematrix_t const M_processus, u32 *v_processus, int n, int transpose)
{

        int nrows = transpose ? M_processus.nrows : M_processus.ncols;

        // fprintf(stderr, "nrows %d\n", n * nrows);

        int size = n * nrows;

        // chaque processeur calcul la matrice qu'il a
        sparse_matrix_vector_product(tmp_processus, &M_processus, v_processus, !transpose);

        // si on calcule la 1ere multiplication
        if (transpose == 1)
        {
                u32 *tmp = malloc(sizeof(*tmp) * size);

                for (int i = 0; i < size; i++)
                {
                        tmp[i] = 0;
                }

                // Create the operation
                MPI_Op operation;

                MPI_Op_create(&sumMod, 1, &operation);

                MPI_Allreduce(tmp_processus, tmp, size, MPI_UINT32_T, operation, MPI_COMM_WORLD);

                MPI_Op_free(&operation);

                for (int i = 0; i < size; i++)
                {
                        tmp_processus[i] = tmp[i];
                }

                free(tmp);
        }
}

// paralell x*x
void computeBlockProduct(u32 *vtAv_processus, u32 *vtAAv_processus, int ncols, u32 *Av_processus, u32 *v_processus)
{

        u32 *vtAv_tmp = malloc((n * n) * sizeof(*vtAv_tmp)); // xt * z

        u32 *vtAAv_tmp = malloc((n * n) * sizeof(*vtAAv_tmp)); // zt * z

        block_dot_products(vtAv_tmp, vtAAv_tmp, ncols, Av_processus, v_processus);

        // Create the operation handle
        MPI_Op operation;
        MPI_Op_create(&sumMod, 1, &operation);

        MPI_Allreduce(vtAv_tmp, vtAv_processus, n * n, MPI_UINT32_T, operation, MPI_COMM_WORLD);

        MPI_Allreduce(vtAAv_tmp, vtAAv_processus, n * n, MPI_UINT32_T, operation, MPI_COMM_WORLD);

        MPI_Op_free(&operation);

        free(vtAv_tmp);
        free(vtAAv_tmp);
}

// gather the final result of V
void gatherFinalV(u32 *v, u32 *v_processus, int processus_ncols, int ncols, int my_rank, int nb_processus)
{

        if (my_rank == 0)
        {

                int recevCounts[nb_processus];

                int displs[nb_processus];

                for (int proc = 0; proc < nb_processus; proc++)
                {

                        recevCounts[proc] = (proc == nb_processus - 1) ? n * (processus_ncols + (ncols % nb_processus)) : n * processus_ncols;
                        displs[proc] = proc * n * processus_ncols;
                }

                MPI_Gatherv(v_processus, n * processus_ncols, MPI_UINT32_T, v, recevCounts, displs, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        }
        else
        {
                MPI_Gatherv(v_processus, n * processus_ncols, MPI_UINT32_T, NULL, NULL, NULL, MPI_UINT32_T, 0, MPI_COMM_WORLD);
        }
}

/************* Sorting functions *************/
void swap(struct sparsematrix_t *M, int u, int i)
{
        u32 temp;
        temp = M->j[u];
        M->j[u] = M->j[i];
        M->j[i] = temp;
        // swap i
        temp = M->i[u];
        M->i[u] = M->i[i];
        M->i[i] = temp;
        // swap x
        temp = M->x[u];
        M->x[u] = M->x[i];
        M->x[i] = temp;
}

// Function that merges the two arrays
void merge(struct sparsematrix_t *M, struct sparsematrix_t arr1, int n1, struct sparsematrix_t arr2, int n2)
{
        M->i = (int *)malloc((n1 + n2) * sizeof(int));

        M->j = (int *)malloc((n1 + n2) * sizeof(int));

        M->x = (u32 *)malloc((n1 + n2) * sizeof(u32));

        int i = 0;
        int u = 0;
        int k;

        for (k = 0; k < n1 + n2; k++)
        {
                if (i >= n1)
                {
                        M->i[k] = arr2.i[u];
                        M->j[k] = arr2.j[u];
                        M->x[k] = arr2.x[u];
                        u++;
                }
                else if (u >= n2)
                {
                        M->i[k] = arr1.i[i];
                        M->j[k] = arr2.j[i];
                        M->x[k] = arr2.x[i];
                        i++;
                }

                // Indices in bounds as i < n1
                // && j < n2
                else if (arr1.j[i] < arr2.j[u])
                {
                        M->i[k] = arr1.i[i];
                        M->j[k] = arr1.j[i];
                        M->x[k] = arr1.x[i];
                        i++;
                }

                // v2[j] <= v1[i]
                else
                {
                        M->i[k] = arr2.i[u];
                        M->j[k] = arr2.j[u];
                        M->x[k] = arr2.x[u];
                        u++;
                }
        }
}

// Function that performs the Quick Sort
void quicksort(struct sparsematrix_t *M_processus, int start, int end)
{
        int pivot, index;

        // Base Case
        if (end <= 1)
                return;

        // Pick pivot and swap with first
        // element Pivot is middle element
        pivot = M_processus->j[start + end / 2];
        swap(M_processus, start, start + end / 2);

        // Partitioning Steps
        index = start;

        // Iterate over the range [start, end]
        for (int i = start + 1; i < start + end; i++)
        {

                // Swap if the element is less
                // than the pivot element
                if (M_processus->j[i] < pivot)
                {
                        index++;
                        swap(M_processus, i, index);
                }
        }

        // Swap the pivot into place
        swap(M_processus, start, index);

        // Recursive Call for sorting
        // of quick sort function
        quicksort(M_processus, start, index - start);
        quicksort(M_processus, index + 1, start + end - index - 1);
}

void sortM(struct sparsematrix_t *M, struct sparsematrix_t *M_processus, int ncols, int my_rank, int nb_processus)
{

        long int size[nb_processus];

        // fprintf(stderr, "gathering  %d\n", my_rank);

        MPI_Allgather(&M_processus->nnz, 1, MPI_LONG, size, 1, MPI_LONG, MPI_COMM_WORLD);

        // Sorting array with quick sort for every
        // chunk as called by process
        quicksort(M_processus, 0, M_processus->nnz);

        int sent = 0;

        for (int step = 1; step < nb_processus; step = 2 * step)
        {
                if (sent != 1 && my_rank % (2 * step) != 0)
                {
                        fprintf(stderr, "Sending %d to %d\n", my_rank, my_rank - step);
                        MPI_Send(M_processus->i, M_processus->nnz, MPI_INT,
                                 my_rank - step, 0,
                                 MPI_COMM_WORLD);

                        MPI_Send(M_processus->j, M_processus->nnz, MPI_INT,
                                 my_rank - step, 1,
                                 MPI_COMM_WORLD);

                        MPI_Send(M_processus->x, M_processus->nnz, MPI_UINT32_T,
                                 my_rank - step, 2,
                                 MPI_COMM_WORLD);

                        sent = 1;
                }

                if (sent != 1 && my_rank + step < nb_processus)
                {
                        fprintf(stderr, "Receiving %d from %d\n", my_rank, my_rank + step);
                        struct sparsematrix_t received;

                        received.i = (int *)malloc(
                            size[my_rank + step] * sizeof(int));
                        received.j = (int *)malloc(
                            size[my_rank + step] * sizeof(int));
                        received.x = (u32 *)malloc(
                            size[my_rank + step] * sizeof(u32));

                        MPI_Recv(received.i, size[my_rank + step],
                                 MPI_INT, my_rank + step, 0,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        MPI_Recv(received.j, size[my_rank + step],
                                 MPI_INT, my_rank + step, 1,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        MPI_Recv(received.x, size[my_rank + step],
                                 MPI_UINT32_T, my_rank + step, 2,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                        merge(M, *M_processus, M_processus->nnz, received, size[my_rank + step]);
                        M_processus->i = M->i;
                        M_processus->j = M->j;
                        M_processus->x = M->x;

                        M_processus->nnz = M_processus->nnz + size[my_rank + step];

                        free(received.i);
                        free(received.j);
                        free(received.x);
                }

                MPI_Allgather(&M_processus->nnz, 1, MPI_LONG, size, 1, MPI_LONG, MPI_COMM_WORLD);
        }

        *M_processus = subdiviseM(*M, nb_processus, my_rank);

        int bloc_size = ncols / nb_processus;

        for (int u = 0; u < M_processus->nnz; u++)
        {
                // dans le cas où y a un padding
                if (M_processus->j[u] >= bloc_size * nb_processus)

                        M_processus->j[u] = M_processus->j[u] % (bloc_size) + bloc_size;

                else
                        M_processus->j[u] = M_processus->j[u] % (bloc_size);
        }
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

/*************************** main functions *********************************/

/* Solve x*M == 0 or M*x == 0 (if transpose == True) */
u32 *block_lanczos(struct sparsematrix_t *M, int n, int my_rank, int nb_processus)
{

        /************* preparations **************/

        int ncols = getCols(*M, my_rank);

        // je subdivise M first
        struct sparsematrix_t M_processus = subdiviseM(*M, nb_processus, my_rank);

        double time = wtime();

        sortM(M, &M_processus, ncols, my_rank, nb_processus);

        if (my_rank == 0)
                printf("Sorting Terminated in %f\n", wtime() - time);

        // Subdivise + sort M

        // je subdivise V
        u32 *v_processus = subdiviseV(ncols, M->nrows, n, nb_processus, my_rank);

        /*********** declarations ***********/
        u32 vtAv_processus[n * n];
        u32 vtAAv_processus[n * n];
        u32 winv[n * n];
        u32 d[n];

        int v_size = M_processus.ncols * n;

        bool stop = false;

        //  division entiere
        long pad = ((ncols + n - 1) / n) * n;

        long Mpad = ((M_processus.nrows + n - 1) / n) * n;

        long block_size_pad = (pad > Mpad ? pad : Mpad) * n;

        u32 *tmp_processus = malloc(sizeof(*tmp_processus) * (block_size_pad));
        u32 *Av_processus = malloc(sizeof(*Av_processus) * (block_size_pad));
        u32 *p = malloc(sizeof(*p) * block_size_pad);

        // will contain the final v
        u32 *v = malloc((ncols * n) * sizeof(*v));

        /*********** initialisations ***********/

        for (int i = 0; i < block_size_pad; i++)
        {
                p[i] = 0;
                tmp_processus[i] = 0;
                Av_processus[i] = 0;
        }

        if (my_rank == 0)
        {
                /* warn the user */
                expected_iterations = 1 + M_processus.nrows / n;
                char human_its[8];
                human_format(human_its, expected_iterations);
                printf("  - Expecting %s iterations\n", human_its);
        }

        start = wtime();

        while (true)
        {
                if (stop_after > 0 && n_iterations == stop_after)
                        break;

                /*tmp = M * v(equivalent en tTD de y = M x)*/
                computeMatrixVectorProduct(tmp_processus, M_processus, v_processus, n, 1);

                /* Av = M *tmp  (equivalent en td de z = Mtransposé *y)*/
                computeMatrixVectorProduct(Av_processus, M_processus, tmp_processus, n, 0);

                // fprintf(stderr, "block block %d\n", my_rank);
                //  bloc bloc B  = vt*Av = (xt * z ) / A = vt*A*Av
                computeBlockProduct(vtAv_processus, vtAAv_processus, M_processus.ncols, Av_processus, v_processus);

                // they will all have the same value of stop coz -> reduce all
                stop = (semi_inverse(vtAv_processus, winv, d) == 0);

                /* check that everything is working ; disable in production */
                // correctness_tests(vtAv_processus, vtAAv_processus, winv, d);

                if (stop)

                        break;

                //   so the new version of v is in tmp
                orthogonalize(v_processus, tmp_processus, p, d, vtAv_processus, vtAAv_processus, winv, M_processus.ncols, Av_processus);

                // we copy tmp in v
                for (long i = 0; i < v_size; i++)
                        v_processus[i] = tmp_processus[i];

                if (my_rank == 0)
                        verbosity();
        }

        if (stop_after < 0 && my_rank == 0)
                final_check(M_processus.ncols, M_processus.nrows, v_processus, tmp_processus);

        printf("  - Terminated in %.1fs after %d iterations\n", wtime() - start, n_iterations);

        free(tmp_processus);
        free(Av_processus);
        free(p);

        gatherFinalV(v, v_processus, M_processus.ncols, ncols, my_rank, nb_processus);

        free(v_processus);

        return v; // x
}

int main(int argc, char **argv)
{

        /* init parallelisation */

        int my_rank;      /* rang du processeur actuel */
        int nb_processus; /* nombre de processeurs */

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nb_processus);

        // command line options -> nothing to change
        // values are the same for every processus
        process_command_line_options(argc, argv);

        // loading the matrix
        struct sparsematrix_t M;

        // processus 0 reecupere la matrice M

        if (my_rank == 0)
        {
                sparsematrix_mm_load(&M, matrix_filename);
                // dans le cas où l'on utilise pas la transposée -> donc we choose left kernel
                // if (right_kernel == 0)
                //         transposeMatrix(&M);
        }

        // coeur du travail
        u32 *kernel = block_lanczos(&M, n, my_rank, nb_processus);

        if (my_rank == 0)
        {

                if (kernel_filename)
                        save_vector_block(kernel_filename, right_kernel ? M.ncols : M.nrows, n, kernel);
                else
                        printf("Not saving result (no --output given)\n");

                free(kernel);
        }

        MPI_Finalize();
        exit(EXIT_SUCCESS);
}
