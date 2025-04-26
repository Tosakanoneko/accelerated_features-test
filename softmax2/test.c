#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define W 8            // Quantization bits
#define LUT_SIZE 7     // LUT size
#define SCALE ((1 << W) - 1)  // Scale for quantization
#define SOFTMAX_TEMP 1.0  // Softmax temperature
#define KPTS_DIM_1 1     // Example dimensions for kpts
#define KPTS_DIM_2 65
#define KPTS_DIM_3 60
#define KPTS_DIM_4 80

// Function to build the LUT
void build_LUT(unsigned char *LUT) {
    for (int i = 0; i < LUT_SIZE; i++) {
        float val = (1.0f / expf(i)) * SCALE;
        LUT[i] = (unsigned char)roundf(val);
    }
}

// Softmax using LUT
void softmax_lut(float (*kpts)[KPTS_DIM_2][KPTS_DIM_3][KPTS_DIM_4], unsigned char *LUT, float (*output)[KPTS_DIM_2][KPTS_DIM_3][KPTS_DIM_4]) {
    float max_val;
    int indices[KPTS_DIM_1][KPTS_DIM_2][KPTS_DIM_3][KPTS_DIM_4];

    // Find the max value for numerical stability
    for (int i = 0; i < KPTS_DIM_1; i++) {
        for (int j = 0; j < KPTS_DIM_2; j++) {
            for (int k = 0; k < KPTS_DIM_3; k++) {
                max_val = -FLT_MAX;
                for (int l = 0; l < KPTS_DIM_4; l++) {
                    if (kpts[i][j][k][l] > max_val) {
                        max_val = kpts[i][j][k][l];
                    }
                }

                // Calculate LUT indices based on max_val - kpts
                for (int l = 0; l < KPTS_DIM_4; l++) {
                    indices[i][j][k][l] = (int)roundf(max_val - kpts[i][j][k][l]);
                    if (indices[i][j][k][l] < 0) indices[i][j][k][l] = 0;
                    if (indices[i][j][k][l] >= LUT_SIZE) indices[i][j][k][l] = LUT_SIZE - 1;

                    // Apply LUT
                    output[i][j][k][l] = LUT[indices[i][j][k][l]];
                }
            }
        }
    }

    // Normalize the output
    for (int i = 0; i < KPTS_DIM_1; i++) {
        for (int j = 0; j < KPTS_DIM_2; j++) {
            for (int k = 0; k < KPTS_DIM_3; k++) {
                float sum = 0.0f;
                for (int l = 0; l < KPTS_DIM_4; l++) {
                    sum += output[i][j][k][l];
                }

                // Normalize
                for (int l = 0; l < KPTS_DIM_4; l++) {
                    output[i][j][k][l] /= sum;
                }
            }
        }
    }
}

// Traditional softmax (using exp)
void softmax_traditional(float (*kpts)[KPTS_DIM_2][KPTS_DIM_3][KPTS_DIM_4], float (*output)[KPTS_DIM_2][KPTS_DIM_3][KPTS_DIM_4]) {
    float max_val;
    float exp_values[KPTS_DIM_1][KPTS_DIM_2][KPTS_DIM_3][KPTS_DIM_4];
    float sum_exp[KPTS_DIM_1][KPTS_DIM_2][KPTS_DIM_3];

    // Find the max value for numerical stability
    for (int i = 0; i < KPTS_DIM_1; i++) {
        for (int j = 0; j < KPTS_DIM_2; j++) {
            for (int k = 0; k < KPTS_DIM_3; k++) {
                max_val = -FLT_MAX;
                for (int l = 0; l < KPTS_DIM_4; l++) {
                    if (kpts[i][j][k][l] > max_val) {
                        max_val = kpts[i][j][k][l];
                    }
                }

                // Compute exp(kpts - max_val)
                for (int l = 0; l < KPTS_DIM_4; l++) {
                    exp_values[i][j][k][l] = expf(kpts[i][j][k][l] - max_val);
                }

                // Compute sum of exp values
                sum_exp[i][j][k] = 0.0f;
                for (int l = 0; l < KPTS_DIM_4; l++) {
                    sum_exp[i][j][k] += exp_values[i][j][k][l];
                }

                // Normalize
                for (int l = 0; l < KPTS_DIM_4; l++) {
                    output[i][j][k][l] = exp_values[i][j][k][l] / sum_exp[i][j][k];
                }
            }
        }
    }
}

int main() {
    printf("hello world\n");
    unsigned char LUT[LUT_SIZE];
    build_LUT(LUT);

    // 动态分配内存
    float (*kpts)[KPTS_DIM_2][KPTS_DIM_3][KPTS_DIM_4] = malloc(KPTS_DIM_1 * KPTS_DIM_2 * KPTS_DIM_3 * KPTS_DIM_4 * sizeof(float));
    float (*scores_traditional)[KPTS_DIM_2][KPTS_DIM_3][KPTS_DIM_4] = malloc(KPTS_DIM_1 * KPTS_DIM_2 * KPTS_DIM_3 * KPTS_DIM_4 * sizeof(float));
    float (*scores_lut)[KPTS_DIM_2][KPTS_DIM_3][KPTS_DIM_4] = malloc(KPTS_DIM_1 * KPTS_DIM_2 * KPTS_DIM_3 * KPTS_DIM_4 * sizeof(float));

    if (kpts == NULL || scores_traditional == NULL || scores_lut == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Example tensor kpts (random values)
    for (int i = 0; i < KPTS_DIM_1; i++) {
        for (int j = 0; j < KPTS_DIM_2; j++) {
            for (int k = 0; k < KPTS_DIM_3; k++) {
                for (int l = 0; l < KPTS_DIM_4; l++) {
                    kpts[i][j][k][l] = (float)rand() / RAND_MAX;
                }
            }
        }
    }

    // Measure time for traditional softmax
    clock_t start_time = clock();
    softmax_traditional(kpts, scores_traditional);
    clock_t end_time = clock();
    double traditional_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Measure time for LUT softmax
    start_time = clock();
    softmax_lut(kpts, LUT, scores_lut);
    end_time = clock();
    double lut_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Print times
    printf("Traditional Softmax Time: %f seconds\n", traditional_time);
    printf("LUT Softmax Time: %f seconds\n", lut_time);

    // Calculate and print the error (mean absolute error)
    float error = 0.0f;
    int count = 0;
    for (int i = 0; i < KPTS_DIM_1; i++) {
        for (int j = 0; j < KPTS_DIM_2; j++) {
            for (int k = 0; k < KPTS_DIM_3; k++) {
                for (int l = 0; l < KPTS_DIM_4; l++) {
                    error += fabsf(scores_traditional[i][j][k][l] - scores_lut[i][j][k][l]);
                    count++;
                }
            }
        }
    }
    error /= count;
    printf("Average Absolute Error: %.6e\n", error);

    // Free dynamically allocated memory
    free(kpts);
    free(scores_traditional);
    free(scores_lut);

    return 0;
}
