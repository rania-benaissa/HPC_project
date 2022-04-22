#define  _POSIX_C_SOURCE 1
#include <inttypes.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <err.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef int64_t i64;
typedef int32_t i32;

u64 prime;
char *matrix_filename;
int nrows;
int ncols;
int d;
u64 seed;

/*************** pseudo-random function (SPECK-like) ********************/

#define ROR(x, r) ((x >> r) | (x << (64 - r)))
#define ROL(x, r) ((x << r) | (x >> (64 - r)))
#define R(x, y, k) (x = ROR(x, 8), x += y, x ^= k, y = ROL(y, 3), y ^= x)
u64 PRF(u64 key, int IV, int i)
{
        u64 y = i + ((u64) IV) << 32;
        u64 x = 0xBaadCafe;
        u64 b = 0xDeadBeef;
        u64 a = seed;
        R(x, y, b);
        for (int i = 0; i < 32; i++) {
                R(a, b, i);
                R(x, y, b);
        }
        return x + i;
}

/******************* utility functions ********************/

double wtime()
{
        struct timeval ts;
        gettimeofday(&ts, NULL);
        return (double) ts.tv_sec + ts.tv_usec / 1e6;
}

void process_command_line_options(int argc, char **argv)
{
        struct option longopts[7] = {
                {"matrix", required_argument, NULL, 'm'},
                {"prime", required_argument, NULL, 'p'},
                {"nrows", required_argument, NULL, 'r'},
                {"ncols", required_argument, NULL, 'c'},
                {"per-row", required_argument, NULL, 'd'},
                {"seed", required_argument, NULL, 's'},
                {NULL, 0, NULL, 0}
        };
        char ch;
        while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
                switch (ch) {
                case 'm':
                        matrix_filename = optarg;
                        break;
                case 'p':
                        prime = atoll(optarg);
                        break;
                case 'r':
                        nrows = atoi(optarg);
                        break;
                case 'c':
                        ncols = atoi(optarg);
                        break;
                case 'd':
                        d = atoi(optarg);
                        break;
                case 's':
                        seed = atoll(optarg);
                        break;
                default:
                        errx(1, "Unknown option\n");
                }
        }
        /* validation */
        if (matrix_filename == NULL || prime == 0 || nrows == 0 || ncols == 0 || d == 0 || seed == 0)
                errx(1, "missing argument");
}

/*************************** main function *********************************/

// hard challenge :
//  - d = 200 entrées par ligne
//  - N = 4M lignes (2^22)

// easy challenge :
//  - d = 200 entrées par ligne
//  - N = 50000 lignes (2^15.5)

int main(int argc, char **argv)
{
        process_command_line_options(argc, argv);
        
        u64 key = seed;
        key = key * 0x1fffffffffffffffull + nrows;
        key = key * 0x1fffffffffffffffull + ncols;
        key = key * 0x1fffffffffffffffull + d;
        key = key * 0x1fffffffffffffffull + prime;


        FILE *f = fopen(matrix_filename, "w");
        if (f == NULL)
                err(1, "fopen");

        fprintf(f, "%%%%MatrixMarket matrix coordinate integer general\n");
        fprintf(f, "%% random matrix with %d nz/row\n", d);
        fprintf(f, "%% seed = %ld\n", seed);
        fprintf(f, "%d %d %d\n", nrows, ncols, d*nrows);

        for (int i = 0; i < nrows; i++) {
                if ((i & 0x3fff) == 0) {
                        printf("\r%.1f%%", (100. * i) / nrows);
                        fflush(stdout);
                }
                for (int k = 0; k < d; k++) {
                        int j = PRF(seed, i, 2*k) % ncols;
                        int x = 1 + (PRF(key, i, 2*k + 1) % (prime - 1));
                        fprintf(f, "%d %d %d\n", i+1, j+1, x);
                }
        }
        fclose(f);
        printf("\n");
        exit(EXIT_SUCCESS);
}