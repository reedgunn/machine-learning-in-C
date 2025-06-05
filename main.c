#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NUM_TRAINING_SETS 4
#define NUM_INPUTS 2
#define NUM_HIDDEN_NODES 20
#define NUM_OUTPUTS 1
#define EPOCHS 100000
#define LEARNING_RATE 0.01


double unif(double lower_bound, double upper_bound) {
    return lower_bound + (double)rand() / RAND_MAX * (upper_bound - lower_bound);
}

typedef struct {
    double hidden_weights[NUM_INPUTS][NUM_HIDDEN_NODES];
    double hidden_bias[NUM_HIDDEN_NODES];
    double output_weights[NUM_HIDDEN_NODES][NUM_OUTPUTS];
    double output_bias[NUM_OUTPUTS];
} Model;

Model init_model() {
    Model res;
    for (int i = 0; i < NUM_INPUTS; i++) {
        for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
            res.hidden_weights[i][j] = unif(-1, 1);
        }
    }
    for (int i = 0; i < NUM_HIDDEN_NODES; i++) {
        res.hidden_bias[i] = unif(-1, 1);
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            res.output_weights[i][j] = unif(-1, 1);
        }
    }
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        res.output_bias[i] = unif(-1, 1);
    }
    return res;
}

typedef struct {
    double inputs[NUM_TRAINING_SETS][NUM_INPUTS];
    double outputs[NUM_TRAINING_SETS][NUM_OUTPUTS];
} TrainingData;

TrainingData init_training_data() {
    TrainingData res = {
        {
            {0.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 1.0}
        },
        {
            {0.0},
            {1.0},
            {1.0},
            {0.0}
        }
    };
    return res;
}

double sigmoid(double x) {
    return 1 / (1 + 1 / exp(x));
}

double sigmoid_derivative(double sigmoid_of_x) {
    return sigmoid_of_x * (1 - sigmoid_of_x);
}

void train_model(Model* model, TrainingData* training_data) {

    double hidden_output[NUM_HIDDEN_NODES];
    double output_output[NUM_OUTPUTS];

    double output_delta[NUM_OUTPUTS];
    double hidden_delta[NUM_HIDDEN_NODES];

    for (int i = 0; i < EPOCHS; i++) {
        for (int j = 0; j < NUM_TRAINING_SETS; j++) {

            for (int k = 0; k < NUM_HIDDEN_NODES; k++) {
                hidden_output[k] = model->hidden_bias[k];
                for (int l = 0; l < NUM_INPUTS; l++) {
                    hidden_output[k] += training_data->inputs[j][l] * model->hidden_weights[l][k];
                }
                hidden_output[k] = sigmoid(hidden_output[k]);
            }
            for (int k = 0; k < NUM_OUTPUTS; k++) {
                output_output[k] = model->output_bias[k];
                for (int l = 0; l < NUM_HIDDEN_NODES; l++) {
                    output_output[k] += hidden_output[l] * model->output_weights[l][k];
                }
                output_output[k] = sigmoid(output_output[k]);
            }

            for (int k = 0; k < NUM_OUTPUTS; k++) {
                output_delta[k] = (training_data->outputs[j][k] - output_output[k]) * sigmoid_derivative(output_output[k]);
            }
            for (int k = 0; k < NUM_HIDDEN_NODES; k++) {
                hidden_delta[k] = 0;
                for (int l = 0; l < NUM_OUTPUTS; l++) {
                    hidden_delta[k] += output_delta[l] * model->output_weights[k][l];
                }
                hidden_delta[k] *= sigmoid_derivative(hidden_output[k]);
            }

            for (int k = 0; k < NUM_OUTPUTS; k++) {
                model->output_bias[k] += LEARNING_RATE * output_delta[k];
                for (int l = 0; l < NUM_HIDDEN_NODES; l++) {
                    model->output_weights[l][k] += LEARNING_RATE * output_delta[k] * hidden_output[l];
                }
            }

            for (int k = 0; k < NUM_HIDDEN_NODES; k++) {
                model->hidden_bias[k] += LEARNING_RATE * hidden_delta[k];
                for (int l = 0; l < NUM_INPUTS; l++) {
                    model->hidden_weights[l][k] += LEARNING_RATE * hidden_delta[k] * training_data->inputs[j][l];
                }
            }

        }
    }
}

void test_model(Model* model, TrainingData* training_data) {
    for (int i = 0; i < NUM_TRAINING_SETS; i++) {

        double hidden_output[NUM_HIDDEN_NODES];
        double output_output[NUM_OUTPUTS];

        for (int j = 0; j < NUM_HIDDEN_NODES; j++) {
            hidden_output[j] = model->hidden_bias[j];
            for (int k = 0; k < NUM_INPUTS; k++) {
                hidden_output[j] += training_data->inputs[i][k] * model->hidden_weights[k][j];
            }
            hidden_output[j] = sigmoid(hidden_output[j]);
        }
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            output_output[j] = model->output_bias[j];
            for (int k = 0; k < NUM_HIDDEN_NODES; k++) {
                output_output[j] += hidden_output[k] * model->output_weights[k][j];
            }
            output_output[j] = sigmoid(output_output[j]);
        }

        printf("Input: [");
        for (int j = 0; j < NUM_INPUTS; j++) {
            printf("%.3f", training_data->inputs[i][j]);
            if (j != NUM_INPUTS - 1) {
                printf(", ");
            }
        }
        printf("]. Actual output: [");
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            printf("%.3f", training_data->outputs[i][j]);
            if (j != NUM_OUTPUTS - 1) {
                printf(", ");
            }
        }
        printf("]. Predicted output: [");
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            printf("%.3f", output_output[j]);
            if (j != NUM_OUTPUTS - 1) {
                printf(", ");
            }
        }
        printf("].\n");
    }
}


int main() {

    srand(time(NULL));

    TrainingData training_data = init_training_data();

    Model model = init_model();

    train_model(&model, &training_data);

    test_model(&model, &training_data);

    return 0;
}
