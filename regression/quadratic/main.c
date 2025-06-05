#include <stdio.h>
#include <math.h>

#define NUM_DATA_POINTS 100


double compute_mean_squared_error(double* a, double* b, double* c, double x[], double y[], long* n) {
    double res = 0;
    for (long i = 0; i < *n; ++i) {
        res += pow(y[i] - (*a + *b * x[i] + *c * pow(x[i], 2)), 2);
    }
    return res / *n;
}


void gradient_descent_step(double* a, double* b, double* c, double x[], double y[], long* n, double* learning_rate) {
    double a_gradient = 0;
    double b_gradient = 0;
    double c_gradient = 0;
    for (long i = 0; i < *n; ++i) {
        a_gradient += 2.0 / *n * (*a + *b * x[i] + *c * pow(x[i], 2) - y[i]);
        b_gradient += 2.0 * x[i] / *n * (*a + *b * x[i] + *c * pow(x[i], 2) - y[i]);
        c_gradient += 2.0 * pow(x[i], 2) / *n * (*a + *b * x[i] + *c * pow(x[i], 2) - y[i]);
    }
    *a -= a_gradient * *learning_rate;
    *b -= b_gradient * *learning_rate;
    *c -= c_gradient * *learning_rate;
}

int main() {

    double a = 0;
    double b = 0;
    double c = 0;

    long iterations = 1e5;

    double learning_rate = 0.0001;

    FILE* file = fopen("data.csv", "r");

    long num_data_points = 100;

    double x[num_data_points];
    double y[num_data_points];

    char header[4];
    fgets(header, sizeof(header), file);
    for (long i = 0; i < num_data_points; ++i) {
        fscanf(file, "%lf,%lf", &x[i], &y[i]);
    }
    fclose(file);


    for (long i = 0; i < iterations; ++i) {
        gradient_descent_step(&a, &b, &c, x, y, &num_data_points, &learning_rate);
    }

    printf("a: %lf\nb: %lf\nc: %lf", a, b, c);

    return 0;
}