/*************************************************************************
	> File Name: subfunction.h
	> Author: mupengcong
	> Mail: mupengcong@lsec.cc.ac.cn
	> Created Time: Mon 31 May 2021 10:08:52 AM CST
 ************************************************************************/
int mmio_read_crd_size(int *m, int *n, int *nnz, char *filename)
{
    int m_tmp, n_tmp;

    int ret_code;
    FILE *f;

    MAT_PTR_TYPE nnz_mtx_report;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_mtx_report;

    return 0;
}

//validate the x by b-A*x
double check_correctness(int n, int *row_ptr, int *col_idx, double *val, double *x, double *b)
{
    double *b_new = (double *)malloc(sizeof(double) * n);
    double *check_b = (double *)malloc(sizeof(double) * n);
    spmv(n, row_ptr, col_idx, val, x, b_new);
    for (int i = 0; i < n; i++)
        check_b[i] = b_new[i] - b[i];
    return vec2norm(check_b, n) / vec2norm(b, n);
}

//store x to a file
void store_x(int n, double *x, char *filename)
{
    FILE *p = fopen(filename, "w");
    fprintf(p, "%d\n", n);
    for (int i = 0; i < n; i++)
        fprintf(p, "%lf\n", x[i]);
    fclose(p);
}

//load right-hand side vector b
void load_b(int n, double *b, char *filename)
{
    FILE *p = fopen(filename, "r");
    int n_right;
    int r = fscanf(p, "%d", &n_right);
    if (n_right != n)
    {
        fclose(p);
        printf("Invalid size of b.\n");
        return;
    }
    for (int i = 0; i < n_right; i++)
        r = fscanf(p, "%lf", &b[i]);
    fclose(p);
}