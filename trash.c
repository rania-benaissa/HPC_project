// if u ever wanna test the matrix : save it !

// if (my_rank == p - 1)
// {

//         FILE *f = fopen("check.mtx", "w");

//         for (long u = 0; u < M_processus.nnz; u++)
//         {
//                 fprintf(f, "%d %d %d\n", M_processus.i[u] + 1, M_processus.j[u] + 1, M_processus.x[u]);
//         }

//         fclose(f);
// }