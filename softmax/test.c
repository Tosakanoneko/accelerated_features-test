#include <stdio.h>
#include <math.h>
#include <time.h>

#define W 8
#define LUT_SIZE 7  // x_q = ceil(ln(2^8 - 1)) ≈ 6, +1 for safety
#define SCALE ((1 << W) - 1) // 255
#define N 50000  // Number of input data
#define ROUNDS 100 // Number of iterations

#define HEIGHT 80  // Image height
#define WIDTH 60   // Image width
#define CHANNELS 65 // Number of channels in kpts

float input[N] = {2.0, 1.0, 0.1, 1.5, 3.0};

// LUT Table
unsigned char LUT[LUT_SIZE];

// Build LUT
void build_LUT() {
    for (int i = 0; i < LUT_SIZE; ++i) {
        float val = (1.0f / expf(i)) * SCALE;
        LUT[i] = (unsigned char)(val + 0.5f);  // Round to nearest integer
    }
}

// Softmax using LUT
void softmax_lut(float *input, float *output, int len) {
    float max_val = input[0];
    for (int i = 1; i < len; ++i)
        if (input[i] > max_val) max_val = input[i];

    int lut_values[N];
    int sum = 0;
    for (int i = 0; i < len; ++i) {
        int index = (int)(max_val - input[i] + 0.5f);
        if (index >= LUT_SIZE) index = LUT_SIZE - 1;
        lut_values[i] = LUT[index];
        sum += lut_values[i];
    }

    for (int i = 0; i < len; ++i) {
        output[i] = (float)lut_values[i] / sum;
    }
}

// Softmax using traditional method (exp)
void softmax_traditional(float *input, float *output, int len) {
    float max_val = input[0];
    for (int i = 1; i < len; ++i)
        if (input[i] > max_val) max_val = input[i];

    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < len; ++i)
        output[i] /= sum;
}

// Apply Softmax to the 4D tensor (image with multiple channels)
void softmax_on_image(float kpts[CHANNELS][HEIGHT][WIDTH], float output[CHANNELS][HEIGHT][WIDTH]) {
    // Apply softmax for each pixel across channels
    for (int h = 0; h < HEIGHT; ++h) {
        for (int w = 0; w < WIDTH; ++w) {
            float pixel[CHANNELS];
            for (int c = 0; c < CHANNELS; ++c) {
                pixel[c] = kpts[c][h][w];
            }
            softmax_traditional(pixel, pixel, CHANNELS);
            // Now take the top 64 values for each pixel
            for (int c = 0; c < 64; ++c) {
                output[c][h][w] = pixel[c];
            }
        }
    }
}

int main() {
    build_LUT();

    // Sample 4D input tensor (kpts)
    float kpts[CHANNELS][HEIGHT][WIDTH] = {{{0}}};  // Initialize with zeros for simplicity
    float output[CHANNELS][HEIGHT][WIDTH] = {{{0}}};

    // Fill kpts with some test data (this would typically be filled with your actual data)
    for (int c = 0; c < CHANNELS; ++c) {
        for (int h = 0; h < HEIGHT; ++h) {
            for (int w = 0; w < WIDTH; ++w) {
                kpts[c][h][w] = (float)(c + h + w);  // Simple test data
            }
        }
    }

    // Run softmax on the image
    clock_t t1 = clock();
    softmax_on_image(kpts, output);
    clock_t t2 = clock();

    printf("Softmax applied to image. Total Time: %ld us\n", (t2 - t1) * 1000000 / CLOCKS_PER_SEC);

    // Print part of the output for verification
    printf("Output at (0,0) for top 64 channels:\n");
    for (int c = 0; c < 64; ++c) {
        printf("output[%d][0][0] = %.6f\n", c, output[c][0][0]);
    }

    return 0;
}
