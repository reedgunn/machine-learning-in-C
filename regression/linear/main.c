#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to compute the Mean Squared Error (MSE)
double compute_mse(double m, double b, double *x, double *y, int n) {
    double error = 0.0;
    for (int i = 0; i < n; i++) {
        double prediction = m * x[i] + b;
        error += (y[i] - prediction) * (y[i] - prediction);
    }
    return error / n;
}

// Function to perform one step of gradient descent
void gradient_descent_step(double *m, double *b, double *x, double *y, int n, double learning_rate) {
    double m_gradient = 0.0;
    double b_gradient = 0.0;
    for (int i = 0; i < n; i++) {
        double prediction = (*m) * x[i] + (*b);
        m_gradient += - (2.0 / n) * x[i] * (y[i] - prediction);
        b_gradient += - (2.0 / n) * (y[i] - prediction);
    }
    *m = *m - learning_rate * m_gradient;
    *b = *b - learning_rate * b_gradient;
}

int main() {
    // Sample data: (x, y)
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {3.0, 4.0, 2.0, 53.0, 6.0};
    int n = sizeof(x) / sizeof(x[0]);
    
    // Hyperparameters
    double learning_rate = 0.01;
    int iterations = 1000;
    
    // Initial guesses for slope (m) and intercept (b)
    double m = 0.0;
    double b = 0.0;
    
    // Gradient descent loop
    for (int i = 0; i < iterations; i++) {
        gradient_descent_step(&m, &b, x, y, n, learning_rate);
        
        // Optionally, print error every 100 iterations
        if (i % 100 == 0) {
            double mse = compute_mse(m, b, x, y, n);
            printf("Iteration %d: m = %f, b = %f, MSE = %f\n", i, m, b, mse);
        }
    }
    
    // Final result
    printf("After %d iterations:\n", iterations);
    printf("Estimated slope (m): %f\n", m);
    printf("Estimated intercept (b): %f\n", b);
    
    return 0;
}
