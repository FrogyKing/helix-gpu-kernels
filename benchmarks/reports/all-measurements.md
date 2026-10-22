# Measured operator results

Source: `benchmark-7a62b743566a490482ef485321ec8b67.json`.

GPU: NVIDIA L4; CUDA 12.6; PyTorch 2.7.1+cu126; Triton 3.3.1.

Dispatch latency includes Python scheduling/output allocation device intervals. Graph replay amortizes ten captured operators. Minimum I/O is analytical.

| Operation | Shape | Dtype | Causal | Backend | Median µs | p95 µs | Graph µs | Speedup | Max abs error |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| softmax | [32, 127] | float32 | False | pytorch | 20.496 | 27.750 | 2.355 | 1.000× | 0 |
| softmax | [32, 127] | float32 | False | triton | 69.120 | 87.998 | 1.946 | 0.297× | 1.49e-08 |
| softmax | [32, 127] | float32 | False | cuda | 21.504 | 33.638 | 2.253 | 0.953× | 7.45e-09 |
| softmax | [32, 127] | float16 | False | pytorch | 21.520 | 40.397 | 2.458 | 1.000× | 0 |
| softmax | [32, 127] | float16 | False | triton | 68.608 | 95.710 | 1.946 | 0.314× | 7.63e-06 |
| softmax | [32, 127] | float16 | False | cuda | 20.480 | 28.229 | 2.202 | 1.051× | 0 |
| softmax | [32, 127] | bfloat16 | False | pytorch | 21.504 | 26.691 | 2.458 | 1.000× | 0 |
| softmax | [32, 127] | bfloat16 | False | triton | 77.312 | 103.629 | 2.048 | 0.278× | 0 |
| softmax | [32, 127] | bfloat16 | False | cuda | 20.480 | 37.274 | 2.253 | 1.050× | 0 |
| softmax | [128, 1024] | float32 | False | pytorch | 21.504 | 37.069 | 4.403 | 1.000× | 0 |
| softmax | [128, 1024] | float32 | False | triton | 70.144 | 116.224 | 2.506 | 0.307× | 7.45e-09 |
| softmax | [128, 1024] | float32 | False | cuda | 21.024 | 34.202 | 3.686 | 1.023× | 3.73e-09 |
| softmax | [128, 1024] | float16 | False | pytorch | 21.504 | 35.840 | 4.813 | 1.000× | 0 |
| softmax | [128, 1024] | float16 | False | triton | 67.584 | 97.485 | 2.304 | 0.318× | 9.54e-07 |
| softmax | [128, 1024] | float16 | False | cuda | 20.480 | 31.590 | 3.686 | 1.050× | 9.54e-07 |
| softmax | [128, 1024] | bfloat16 | False | pytorch | 21.504 | 32.018 | 4.864 | 1.000× | 0 |
| softmax | [128, 1024] | bfloat16 | False | triton | 76.256 | 108.466 | 2.251 | 0.282× | 0 |
| softmax | [128, 1024] | bfloat16 | False | cuda | 20.480 | 32.461 | 3.789 | 1.050× | 9.54e-07 |
| softmax | [32, 8193] | float32 | False | pytorch | 21.504 | 30.822 | 5.120 | 1.000× | 0 |
| softmax | [32, 8193] | float32 | False | triton | 68.608 | 95.334 | 4.506 | 0.313× | 9.31e-10 |
| softmax | [32, 8193] | float32 | False | cuda | 22.528 | 25.139 | 12.493 | 0.955× | 6.98e-10 |
| softmax | [32, 8193] | float16 | False | pytorch | 21.504 | 24.576 | 5.018 | 1.000× | 0 |
| softmax | [32, 8193] | float16 | False | triton | 67.584 | 89.088 | 3.686 | 0.318× | 2.38e-07 |
| softmax | [32, 8193] | float16 | False | cuda | 22.528 | 34.970 | 12.595 | 0.955× | 2.38e-07 |
| softmax | [32, 8193] | bfloat16 | False | pytorch | 21.504 | 31.898 | 5.069 | 1.000× | 0 |
| softmax | [32, 8193] | bfloat16 | False | triton | 76.800 | 107.059 | 3.686 | 0.280× | 0 |
| softmax | [32, 8193] | bfloat16 | False | cuda | 23.552 | 29.440 | 12.595 | 0.913× | 0 |
| rmsnorm | [32, 127] | float32 | False | pytorch | 80.896 | 130.048 | 8.344 | 1.000× | 0 |
| rmsnorm | [32, 127] | float32 | False | pytorch_materialized | 100.352 | 127.218 | 8.346 | 0.806× | 0 |
| rmsnorm | [32, 127] | float32 | False | triton | 78.848 | 123.034 | 1.741 | 1.026× | 9.54e-07 |
| rmsnorm | [32, 127] | float32 | False | cuda | 21.472 | 31.291 | 2.048 | 3.768× | 9.54e-07 |
| rmsnorm | [32, 127] | float16 | False | pytorch | 104.448 | 135.253 | 12.698 | 1.000× | 0 |
| rmsnorm | [32, 127] | float16 | False | pytorch_materialized | 141.312 | 165.171 | 12.595 | 0.739× | 0 |
| rmsnorm | [32, 127] | float16 | False | triton | 79.872 | 99.013 | 1.792 | 1.308× | 0 |
| rmsnorm | [32, 127] | float16 | False | cuda | 21.504 | 34.691 | 2.099 | 4.857× | 0 |
| rmsnorm | [32, 127] | bfloat16 | False | pytorch | 105.984 | 135.219 | 13.210 | 1.000× | 0 |
| rmsnorm | [32, 127] | bfloat16 | False | pytorch_materialized | 141.312 | 170.650 | 12.698 | 0.750× | 0 |
| rmsnorm | [32, 127] | bfloat16 | False | triton | 92.160 | 121.037 | 1.741 | 1.150× | 0 |
| rmsnorm | [32, 127] | bfloat16 | False | cuda | 20.512 | 32.256 | 2.048 | 5.167× | 0 |
| rmsnorm | [128, 1024] | float32 | False | pytorch | 77.824 | 107.213 | 10.496 | 1.000× | 0 |
| rmsnorm | [128, 1024] | float32 | False | pytorch_materialized | 96.768 | 122.522 | 10.547 | 0.804× | 0 |
| rmsnorm | [128, 1024] | float32 | False | triton | 77.824 | 114.893 | 2.406 | 1.000× | 1.43e-06 |
| rmsnorm | [128, 1024] | float32 | False | cuda | 21.472 | 33.624 | 3.379 | 3.624× | 1.43e-06 |
| rmsnorm | [128, 1024] | float16 | False | pytorch | 103.936 | 133.888 | 15.565 | 1.000× | 0 |
| rmsnorm | [128, 1024] | float16 | False | pytorch_materialized | 135.680 | 169.267 | 15.974 | 0.766× | 0 |
| rmsnorm | [128, 1024] | float16 | False | triton | 81.920 | 141.534 | 2.099 | 1.269× | 7.63e-06 |
| rmsnorm | [128, 1024] | float16 | False | cuda | 21.504 | 38.400 | 3.277 | 4.833× | 0.000244 |
| rmsnorm | [128, 1024] | bfloat16 | False | pytorch | 107.520 | 148.070 | 15.974 | 1.000× | 0 |
| rmsnorm | [128, 1024] | bfloat16 | False | pytorch_materialized | 146.944 | 169.062 | 16.077 | 0.732× | 0 |
| rmsnorm | [128, 1024] | bfloat16 | False | triton | 92.160 | 121.976 | 2.048 | 1.167× | 0 |
| rmsnorm | [128, 1024] | bfloat16 | False | cuda | 21.504 | 42.394 | 3.226 | 5.000× | 0 |
| rmsnorm | [32, 8193] | float32 | False | pytorch | 77.312 | 99.328 | 12.186 | 1.000× | 0 |
| rmsnorm | [32, 8193] | float32 | False | pytorch_materialized | 99.328 | 127.299 | 12.186 | 0.778× | 0 |
| rmsnorm | [32, 8193] | float32 | False | triton | 78.848 | 106.632 | 5.274 | 0.981× | 1.91e-06 |
| rmsnorm | [32, 8193] | float32 | False | cuda | 21.504 | 33.024 | 12.083 | 3.595× | 1.91e-06 |
| rmsnorm | [32, 8193] | float16 | False | pytorch | 107.008 | 136.294 | 17.408 | 1.000× | 0 |
| rmsnorm | [32, 8193] | float16 | False | pytorch_materialized | 140.800 | 171.213 | 18.432 | 0.760× | 0 |
| rmsnorm | [32, 8193] | float16 | False | triton | 78.320 | 100.557 | 4.147 | 1.366× | 0.000488 |
| rmsnorm | [32, 8193] | float16 | False | cuda | 22.528 | 28.416 | 12.186 | 4.750× | 0.00195 |
| rmsnorm | [32, 8193] | bfloat16 | False | pytorch | 106.032 | 131.797 | 17.664 | 1.000× | 0 |
| rmsnorm | [32, 8193] | bfloat16 | False | pytorch_materialized | 141.824 | 162.398 | 18.637 | 0.748× | 0 |
| rmsnorm | [32, 8193] | bfloat16 | False | triton | 94.208 | 126.413 | 4.403 | 1.126× | 0 |
| rmsnorm | [32, 8193] | bfloat16 | False | cuda | 22.512 | 30.106 | 12.237 | 4.710× | 0 |
| gemm | [64, 96, 80] | float32 | False | pytorch | 34.816 | 61.184 | 6.144 | 1.000× | 5.72e-06 |
| gemm | [64, 96, 80] | float32 | False | triton | 77.824 | 158.822 | 3.226 | 0.447× | 9.54e-06 |
| gemm | [64, 96, 80] | float32 | False | cuda | 20.528 | 34.253 | 3.891 | 1.696× | 9.54e-06 |
| gemm | [64, 96, 80] | float32 | False | cuda_naive | 23.552 | 37.171 | 5.939 | 1.478× | 9.54e-06 |
| gemm | [64, 96, 80] | float16 | False | pytorch | 31.744 | 51.792 | 2.714 | 1.000× | 0.00391 |
| gemm | [64, 96, 80] | float16 | False | triton | 73.216 | 98.150 | 2.355 | 0.434× | 0.00391 |
| gemm | [64, 96, 80] | float16 | False | cuda | 28.160 | 41.370 | 3.891 | 1.127× | 0.00391 |
| gemm | [64, 96, 80] | float16 | False | cuda_naive | 23.552 | 29.338 | 6.246 | 1.348× | 0.00391 |
| gemm | [64, 96, 80] | bfloat16 | False | pytorch | 30.720 | 41.472 | 3.021 | 1.000× | 0 |
| gemm | [64, 96, 80] | bfloat16 | False | triton | 88.064 | 115.456 | 2.150 | 0.349× | 0 |
| gemm | [64, 96, 80] | bfloat16 | False | cuda | 20.480 | 22.528 | 3.789 | 1.500× | 0 |
| gemm | [64, 96, 80] | bfloat16 | False | cuda_naive | 23.536 | 27.187 | 6.144 | 1.305× | 0 |
| gemm | [256, 256, 256] | float32 | False | pytorch | 28.672 | 47.206 | 8.037 | 1.000× | 1.72e-05 |
| gemm | [256, 256, 256] | float32 | False | triton | 76.288 | 108.339 | 8.141 | 0.376× | 4.58e-05 |
| gemm | [256, 256, 256] | float32 | False | cuda | 29.696 | 38.298 | 19.866 | 0.966× | 4.58e-05 |
| gemm | [256, 256, 256] | float32 | False | cuda_naive | 40.960 | 43.571 | 27.290 | 0.700× | 4.58e-05 |
| gemm | [256, 256, 256] | float16 | False | pytorch | 31.744 | 39.936 | 4.301 | 1.000× | 0.0156 |
| gemm | [256, 256, 256] | float16 | False | triton | 75.776 | 96.358 | 3.277 | 0.419× | 0.0156 |
| gemm | [256, 256, 256] | float16 | False | cuda | 29.696 | 33.434 | 19.661 | 1.069× | 0.0156 |
| gemm | [256, 256, 256] | float16 | False | cuda_naive | 40.960 | 46.746 | 28.262 | 0.775× | 0.0156 |
| gemm | [256, 256, 256] | bfloat16 | False | pytorch | 31.744 | 51.968 | 5.427 | 1.000× | 0.125 |
| gemm | [256, 256, 256] | bfloat16 | False | triton | 139.776 | 338.074 | 3.222 | 0.227× | 0.125 |
| gemm | [256, 256, 256] | bfloat16 | False | cuda | 30.720 | 36.403 | 19.763 | 1.033× | 0.0625 |
| gemm | [256, 256, 256] | bfloat16 | False | cuda_naive | 41.472 | 58.059 | 27.902 | 0.765× | 0.0625 |
| gemm | [128, 768, 3072] | float32 | False | pytorch | 75.776 | 87.040 | 53.453 | 1.000× | 9.16e-05 |
| gemm | [128, 768, 3072] | float32 | False | triton | 140.800 | 397.619 | 68.659 | 0.538× | 0.000145 |
| gemm | [128, 768, 3072] | float32 | False | cuda | 273.408 | 278.715 | 259.482 | 0.277× | 0.000145 |
| gemm | [128, 768, 3072] | float32 | False | cuda_naive | 384.000 | 390.810 | 351.642 | 0.197× | 0.000145 |
| gemm | [128, 768, 3072] | float16 | False | pytorch | 30.720 | 52.429 | 11.162 | 1.000× | 0.0625 |
| gemm | [128, 768, 3072] | float16 | False | triton | 81.920 | 102.144 | 21.709 | 0.375× | 0.0625 |
| gemm | [128, 768, 3072] | float16 | False | cuda | 272.864 | 277.606 | 259.072 | 0.113× | 0.0625 |
| gemm | [128, 768, 3072] | float16 | False | cuda_naive | 369.120 | 377.702 | 352.307 | 0.083× | 0.0625 |
| gemm | [128, 768, 3072] | bfloat16 | False | pytorch | 30.224 | 43.096 | 10.957 | 1.000× | 0.25 |
| gemm | [128, 768, 3072] | bfloat16 | False | triton | 98.816 | 141.773 | 20.890 | 0.306× | 0.25 |
| gemm | [128, 768, 3072] | bfloat16 | False | cuda | 273.408 | 276.019 | 259.277 | 0.111× | 0.25 |
| gemm | [128, 768, 3072] | bfloat16 | False | cuda_naive | 369.136 | 386.714 | 354.304 | 0.082× | 0.25 |
| attention | [1, 2, 65, 32] | float32 | False | pytorch | 39.936 | 56.115 | 14.336 | 1.000× | 1.01e-06 |
| attention | [1, 2, 65, 32] | float32 | False | pytorch_materialized | 118.784 | 154.266 | 21.094 | 0.336× | 0 |
| attention | [1, 2, 65, 32] | float32 | False | triton | 94.208 | 113.186 | 6.758 | 0.424× | 2.98e-07 |
| attention | [1, 2, 65, 32] | float32 | False | cuda | 24.576 | 33.946 | 13.618 | 1.625× | 4.17e-07 |
| attention | [1, 2, 65, 32] | float32 | True | pytorch | 36.880 | 52.224 | 14.131 | 1.000× | 1.07e-06 |
| attention | [1, 2, 65, 32] | float32 | True | pytorch_materialized | 223.744 | 253.168 | 26.829 | 0.165× | 0 |
| attention | [1, 2, 65, 32] | float32 | True | triton | 90.128 | 115.269 | 6.554 | 0.409× | 5.36e-07 |
| attention | [1, 2, 65, 32] | float32 | True | cuda | 23.552 | 36.352 | 12.698 | 1.566× | 5.96e-07 |
| attention | [1, 2, 65, 32] | float16 | False | pytorch | 42.496 | 61.952 | 4.915 | 1.000× | 0.000488 |
| attention | [1, 2, 65, 32] | float16 | False | pytorch_materialized | 251.392 | 304.640 | 27.853 | 0.169× | 0 |
| attention | [1, 2, 65, 32] | float16 | False | triton | 91.136 | 116.378 | 4.813 | 0.466× | 6.1e-05 |
| attention | [1, 2, 65, 32] | float16 | False | cuda | 24.576 | 27.648 | 14.029 | 1.729× | 0.000122 |
| attention | [1, 2, 65, 32] | float16 | True | pytorch | 41.984 | 59.955 | 5.018 | 1.000× | 0.000977 |
| attention | [1, 2, 65, 32] | float16 | True | pytorch_materialized | 278.016 | 327.168 | 33.792 | 0.151× | 0 |
| attention | [1, 2, 65, 32] | float16 | True | triton | 88.592 | 124.482 | 4.198 | 0.474× | 0.000122 |
| attention | [1, 2, 65, 32] | float16 | True | cuda | 23.552 | 30.259 | 13.312 | 1.783× | 0.000122 |
| attention | [1, 2, 65, 32] | bfloat16 | False | pytorch | 41.984 | 61.053 | 5.014 | 1.000× | 0.00391 |
| attention | [1, 2, 65, 32] | bfloat16 | False | pytorch_materialized | 179.712 | 223.078 | 28.058 | 0.234× | 0 |
| attention | [1, 2, 65, 32] | bfloat16 | False | triton | 109.056 | 142.746 | 4.813 | 0.385× | 7.63e-06 |
| attention | [1, 2, 65, 32] | bfloat16 | False | cuda | 24.576 | 26.163 | 13.619 | 1.708× | 7.63e-06 |
| attention | [1, 2, 65, 32] | bfloat16 | True | pytorch | 41.984 | 65.280 | 5.120 | 1.000× | 0.00781 |
| attention | [1, 2, 65, 32] | bfloat16 | True | pytorch_materialized | 298.496 | 321.667 | 34.099 | 0.141× | 0 |
| attention | [1, 2, 65, 32] | bfloat16 | True | triton | 113.168 | 141.875 | 4.301 | 0.371× | 0 |
| attention | [1, 2, 65, 32] | bfloat16 | True | cuda | 23.024 | 26.691 | 12.800 | 1.823× | 0.000122 |
| attention | [1, 4, 256, 64] | float32 | False | pytorch | 54.272 | 60.058 | 32.768 | 1.000× | 1.13e-06 |
| attention | [1, 4, 256, 64] | float32 | False | pytorch_materialized | 154.112 | 221.542 | 44.390 | 0.352× | 0 |
| attention | [1, 4, 256, 64] | float32 | False | triton | 119.808 | 150.938 | 45.568 | 0.453× | 4.47e-07 |
| attention | [1, 4, 256, 64] | float32 | False | cuda | 187.904 | 190.514 | 174.387 | 0.289× | 6.56e-07 |
| attention | [1, 4, 256, 64] | float32 | True | pytorch | 55.296 | 64.682 | 32.870 | 1.000× | 1.67e-06 |
| attention | [1, 4, 256, 64] | float32 | True | pytorch_materialized | 229.376 | 254.515 | 52.837 | 0.241× | 0 |
| attention | [1, 4, 256, 64] | float32 | True | triton | 104.960 | 133.069 | 32.512 | 0.527× | 5.36e-07 |
| attention | [1, 4, 256, 64] | float32 | True | cuda | 134.144 | 138.342 | 121.242 | 0.412× | 6.26e-07 |
| attention | [1, 4, 256, 64] | float16 | False | pytorch | 43.008 | 55.603 | 8.704 | 1.000× | 0.000244 |
| attention | [1, 4, 256, 64] | float16 | False | pytorch_materialized | 184.320 | 218.010 | 51.810 | 0.233× | 0 |
| attention | [1, 4, 256, 64] | float16 | False | triton | 89.088 | 121.600 | 14.029 | 0.483× | 0.000122 |
| attention | [1, 4, 256, 64] | float16 | False | cuda | 188.384 | 199.373 | 173.312 | 0.228× | 0.000122 |
| attention | [1, 4, 256, 64] | float16 | True | pytorch | 43.008 | 65.558 | 9.216 | 1.000× | 0.000977 |
| attention | [1, 4, 256, 64] | float16 | True | pytorch_materialized | 290.816 | 334.797 | 60.262 | 0.148× | 0 |
| attention | [1, 4, 256, 64] | float16 | True | triton | 94.240 | 125.850 | 11.366 | 0.456× | 0.000244 |
| attention | [1, 4, 256, 64] | float16 | True | cuda | 133.104 | 139.504 | 121.648 | 0.323× | 0.000244 |
| attention | [1, 4, 256, 64] | bfloat16 | False | pytorch | 43.008 | 62.874 | 8.806 | 1.000× | 0.00195 |
| attention | [1, 4, 256, 64] | bfloat16 | False | pytorch_materialized | 193.536 | 225.843 | 51.866 | 0.222× | 0 |
| attention | [1, 4, 256, 64] | bfloat16 | False | triton | 110.592 | 130.611 | 13.978 | 0.389× | 0.000488 |
| attention | [1, 4, 256, 64] | bfloat16 | False | cuda | 185.344 | 188.979 | 171.722 | 0.232× | 0.000488 |
| attention | [1, 4, 256, 64] | bfloat16 | True | pytorch | 43.008 | 65.075 | 9.114 | 1.000× | 0.00781 |
| attention | [1, 4, 256, 64] | bfloat16 | True | pytorch_materialized | 281.088 | 317.389 | 60.109 | 0.153× | 0 |
| attention | [1, 4, 256, 64] | bfloat16 | True | triton | 135.168 | 186.522 | 11.418 | 0.318× | 0.000977 |
| attention | [1, 4, 256, 64] | bfloat16 | True | cuda | 133.120 | 145.715 | 122.573 | 0.323× | 0.000122 |
| quantized_linear | [1, 128, 256] | float32 | False | pytorch | 84.480 | 131.342 | 5.632 | 1.000× | 0 |
| quantized_linear | [1, 128, 256] | float32 | False | pytorch_unquantized | 31.792 | 50.278 | 2.253 | 2.657× | 0.239 |
| quantized_linear | [1, 128, 256] | float32 | False | triton | 84.480 | 109.056 | 6.758 | 1.000× | 1.14e-05 |
| quantized_linear | [1, 128, 256] | float32 | False | cuda | 21.504 | 35.072 | 4.608 | 3.929× | 1.53e-05 |
| quantized_linear | [1, 128, 256] | float16 | False | pytorch | 111.104 | 137.450 | 7.834 | 1.000× | 0 |
| quantized_linear | [1, 128, 256] | float16 | False | pytorch_unquantized | 32.768 | 49.818 | 2.253 | 3.391× | 0.174 |
| quantized_linear | [1, 128, 256] | float16 | False | triton | 81.920 | 115.866 | 7.474 | 1.356× | 0.000977 |
| quantized_linear | [1, 128, 256] | float16 | False | cuda | 20.512 | 23.091 | 4.506 | 5.417× | 0.000977 |
| quantized_linear | [1, 128, 256] | bfloat16 | False | pytorch | 113.152 | 172.603 | 7.885 | 1.000× | 0 |
| quantized_linear | [1, 128, 256] | bfloat16 | False | pytorch_unquantized | 32.768 | 53.619 | 2.251 | 3.453× | 0.25 |
| quantized_linear | [1, 128, 256] | bfloat16 | False | triton | 92.144 | 123.546 | 7.475 | 1.228× | 0 |
| quantized_linear | [1, 128, 256] | bfloat16 | False | cuda | 20.944 | 29.099 | 4.506 | 5.403× | 0 |
| quantized_linear | [32, 512, 768] | float32 | False | pytorch | 94.208 | 129.997 | 16.077 | 1.000× | 0 |
| quantized_linear | [32, 512, 768] | float32 | False | pytorch_unquantized | 37.888 | 73.165 | 9.523 | 2.486× | 0.773 |
| quantized_linear | [32, 512, 768] | float32 | False | triton | 87.552 | 111.923 | 22.016 | 1.076× | 8.01e-05 |
| quantized_linear | [32, 512, 768] | float32 | False | cuda | 28.672 | 40.550 | 19.149 | 3.286× | 8.39e-05 |
| quantized_linear | [32, 512, 768] | float16 | False | pytorch | 116.736 | 143.923 | 19.149 | 1.000× | 0 |
| quantized_linear | [32, 512, 768] | float16 | False | pytorch_unquantized | 34.272 | 47.514 | 4.198 | 3.406× | 0.703 |
| quantized_linear | [32, 512, 768] | float16 | False | triton | 91.136 | 127.386 | 24.986 | 1.281× | 0.0312 |
| quantized_linear | [32, 512, 768] | float16 | False | cuda | 27.648 | 30.259 | 18.432 | 4.222× | 0.0156 |
| quantized_linear | [32, 512, 768] | bfloat16 | False | pytorch | 120.320 | 153.139 | 19.251 | 1.000× | 0 |
| quantized_linear | [32, 512, 768] | bfloat16 | False | pytorch_unquantized | 33.808 | 54.491 | 5.120 | 3.559× | 0.75 |
| quantized_linear | [32, 512, 768] | bfloat16 | False | triton | 102.912 | 144.333 | 25.088 | 1.169× | 0.0312 |
| quantized_linear | [32, 512, 768] | bfloat16 | False | cuda | 28.672 | 38.722 | 18.278 | 4.196× | 6.1e-05 |

## Validation failures

None.
