| backend | precision | iterations | final LL | abs. LL difference | mean matched corr | min matched corr | Amari distance | runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fortran (reference) | float64 | 100 | -3.411274 | | | | | 10.0 |
| PyTorch | float64 | 100 | -3.411003 | 0.000271 | 0.9991 | 0.9918 | 0.0038 | 18.1 |
| NumPy | float64 | 100 | -3.411003 | 0.000272 | 0.9991 | 0.9917 | 0.0038 | 32.6 |
| MLX | float32 | 100 | -3.410998 | 0.000276 | 0.9991 | 0.9917 | 0.0038 | 3.0 |
