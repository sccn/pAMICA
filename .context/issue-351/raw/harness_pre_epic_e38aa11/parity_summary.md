| backend | precision | iterations | final LL | abs. LL difference | mean matched corr | min matched corr | Amari distance | runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fortran (reference) | float64 | 100 | -3.411274 | | | | | 10.1 |
| PyTorch | float64 | 100 | -3.411245 | 0.000029 | 0.9992 | 0.9935 | 0.0037 | 17.9 |
| NumPy | float64 | 100 | -3.411246 | 0.000029 | 0.9992 | 0.9934 | 0.0037 | 32.5 |
| MLX | float32 | 100 | -3.411242 | 0.000032 | 0.9992 | 0.9935 | 0.0037 | 3.2 |
