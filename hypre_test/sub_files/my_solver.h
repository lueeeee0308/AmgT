/*
A simple serial CG iterative method to solve a linear system
@Code version: 1.0
@Update date: 2021/5/17
@Author: Dechuang Yang,Haocheng Lian
*/
// Multiply a csr matrix with a vector x, and get the resulting vector y
void spmv(int n, int *row_ptr, int *col_idx, double *val, double *x, double *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            y[i] += val[j] * x[col_idx[j]];
    }
}

// Calculate the 2-norm of a vector
double vec2norm(double *x, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];
    return sqrt(sum);
}

// Compute dot product of two vectors, and return the result
double dotproduct(double *x1, double *x2, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += x1[i] * x2[i];
    return sum;
}

// Solve a linear system by using a simple CG iterative method
void my_solver(int n, int *row_ptr, int *col_idx, double *val,
               double *x, double *b, int *iter, double tolerance)
{
    int maxiter = 1000;
    memset(x, 0, sizeof(double) * n);
    double *r = (double *)malloc(sizeof(double) * n);
    double *y = (double *)malloc(sizeof(double) * n);
    double *p = (double *)malloc(sizeof(double) * n);
    double *q = (double *)malloc(sizeof(double) * n);
    *iter = 0;
    double norm = 0.0;
    double rho = 0.0;
    double rho_1 = 0.0;
    double error = 0.0;

    spmv(n, row_ptr, col_idx, val, x, y);
    for (int i = 0; i < n; i++)
        r[i] = b[i] - y[i];

    do
    {
        rho = dotproduct(r, r, n);
        if (*iter == 0)
        {
            for (int i = 0; i < n; i++)
                p[i] = r[i];
        }
        else
        {
            double beta = rho / rho_1;
            for (int i = 0; i < n; i++)
                p[i] = r[i] + beta * p[i];
        }

        spmv(n, row_ptr, col_idx, val, p, q);
        double alpha = rho / dotproduct(p, q, n);

        for (int i = 0; i < n; i++)
            x[i] += alpha * p[i];
        for (int i = 0; i < n; i++)
            r[i] += -alpha * q[i];

        rho_1 = rho;
        error = vec2norm(r, n);
        *iter += 1;

        if (error < tolerance)
            break;
    } while (*iter < maxiter);

    free(r);
    free(y);
    free(p);
    free(q);
}