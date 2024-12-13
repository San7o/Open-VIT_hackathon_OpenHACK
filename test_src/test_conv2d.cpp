#include "../include/conv2d.h"

#include <iostream>

using namespace std;

int main() {
    cout << "Test 2D Convolution" << endl;

    vit_float k_data[4*2*3*3] = {
         21.652,  45.315,  17.144,
         80.826,  -3.726, -64.446,
          8.782,  84.980,  82.643,

         -2.662, -70.105,  50.542,
         18.906, -88.199, -93.445,
        -61.458, -60.664,   0.305,



         54.202, -60.861,  -7.207,
        -56.091,  16.637,  10.484,
         -3.454,  13.944, -65.273,

        -68.912, -11.488,   2.296,
         78.929,  17.744, -68.899,
        -93.467, -37.562, -12.479,



         -3.002,  99.734,  15.456,
        -71.625, -91.314, -93.800,
         80.448, -93.347,  35.666,

        -68.885,  57.565, -37.414,
        -37.276, -13.045, -83.517,
         66.691, -66.863,  45.049,



         -3.179, -89.956, -20.579,
        -44.226, -84.967,  93.510,
         57.031, -78.537, -60.847,

        -85.909, -80.929, -99.998,
         -3.576, -56.245, -76.798,
         10.084, -68.542,  30.343
    };
    PictureBatch k(k_data, 4*2*3*3, 4, 2, 3, 3);
    cout << "### k" << endl;
    k.print();

    vit_float b_data[4] = {-5.247, -10.884, -60.649, 51.270};
    RowVector b(b_data, 4);
    cout << "### b" << endl;
    b.print();

    vit_float x_data[3*2*5*9] = {
         19.964,  24.195,  42.694, -41.003, -72.493,   5.917, -16.969,  78.681, -54.181,
         -7.477,  -6.084, -65.014, -87.909, -27.730, -33.484,  21.848,  28.560,  20.049,
        -26.980,  49.665,  -4.151,  83.432,  32.653, -62.198,  38.191,  11.448,  19.779,
         43.249, -39.433,  36.220,  33.256, -72.743, -86.477, -29.260, -74.915,  11.887,
        -48.205,  24.720,  82.635,  60.938,  53.914, -75.690, -39.047,  59.496, -17.582,

          6.198,  11.976,  80.124,  63.922,  29.019, -12.138, -62.941, -61.756,  45.375,
        -19.520,   0.617, -92.871, -20.771, -60.114,  20.879,  10.129,  92.993,  34.461,
         82.739,  28.903,  55.451,  69.783,  43.902,  69.240,  84.777, -62.107, -88.122,
        -91.402, -63.835,  50.363,  17.952, -47.124,  86.356, -41.309,  34.074, -80.352,
         13.787,  89.644, -50.684,  56.895, -85.997, -40.031, -82.862, -16.072,  81.905,



        -99.403,  16.150, -17.495,  51.594, -92.912,  85.143,  22.674, -91.548, -31.211,
         69.954, -73.485,   9.040, -24.594, -55.993, -25.652, -83.310, -78.177,  83.057,
         40.225,  74.855, -14.801, -89.419,  58.663,  52.652,  33.604,  72.329,  96.939,
         29.034, -93.975,  30.269,  20.351, -26.963,  -1.388, -24.659, -37.551,  71.145,
        -31.037, -58.885,  80.083, -52.360,  70.986, -90.076,  -4.383, -96.724,  33.267,

         91.011, -50.023,  -6.377, -99.664, -73.396,  16.040, -73.999,  -4.795, -47.284,
         92.164,  91.381,  18.770,  -0.636,  52.467, -58.033, -71.671, -25.992, -67.827,
        -33.288, -13.392,  87.979, -40.489,  10.212,  96.822,  28.417,  37.811, -18.781,
        -13.976,  90.167,  27.288, -28.822, -30.991,  11.963,  68.684,  57.392,  36.294,
         -5.004, -37.225,  58.562,  36.162,  64.875,  14.040, -96.973, -38.491, -33.052,



        -73.190, -33.611, -23.165,  -4.367, -62.287,  85.059, -76.504, -45.370,  34.950,
         48.099,  55.276, -44.412,  62.206,  17.868,  34.038,  33.067,  42.795, -98.962,
        -42.783,  64.009,  17.278,   5.175,  -9.595, -18.169,  89.434, -84.530,  18.663,
         83.643,  32.832, -65.174, -88.235,  90.118, -87.308,  25.197, -21.393,  70.311,
         18.539, -14.786, -40.477,  -2.320, -31.636,  94.613, -21.736, -19.172, -66.005,

         14.998, -19.578, -59.120, -10.443,  76.346,   9.216,  21.807, -55.002, -10.979,
         15.088, -16.755,  90.127, -10.002, -53.753, -38.352, -96.375,  -1.640,  99.576,
        -83.132, -23.159,  64.604, -94.931,  81.507, -75.784, -51.779,  55.452, -64.247,
        -99.559, -45.633, -22.290,  42.457, -29.668, -11.325,  51.994,  64.720,  57.700,
         18.403, -29.482,  -7.431,  15.015, -24.908,  83.084,  88.144, -47.516,  40.005
    };
    PictureBatch x(x_data, 3*2*5*9, 3, 2, 5, 9);
    cout << "### x" << endl;
    x.print();

    Conv2d c2d(2, 4, 3, 3, 2, 2, true);
    c2d.move_kernel(k);
    c2d.move_bias(b);

    PictureBatch y;
    c2d.forward(x, y);
    cout << "### y = conv(x) stride 2,2" << endl;
    y.print();

    return 0;
}
