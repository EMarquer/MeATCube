
# Benchmark report


## balance+scale dataset
|                    | deletion rate   |   cb_size |   step | F1      | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:--------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |       375 |      0 | 55.83%  | 56.80%     | 9.60%                  | 76.80%         |
| best MeATCube F1   | 80.80%          |        72 |    303 | 100.00% | 100.00%    | 51.20%                 | 71.20%         |
| best MeATCube acc. | 80.80%          |        72 |    303 | 100.00% | 100.00%    | 51.20%                 | 71.20%         |
| best KNN acc.      | 89.87%          |        38 |    337 | 87.74%  | 94.40%     | 91.20%                 | 68.80%         |
| best 1NN acc.      | 10.13%          |       337 |     38 | 61.63%  | 64.00%     | 9.60%                  | 81.60%         |

![](results/figs/balance+scale.png)


## breast+cancer+wisconsin+diagnostic dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |       341 |      0 | 90.48% | 91.23%     | 91.23%                 | 93.86%         |
| best MeATCube F1   | 79.18%          |        71 |    270 | 94.45% | 94.74%     | 86.84%                 | 54.39%         |
| best MeATCube acc. | 79.18%          |        71 |    270 | 94.45% | 94.74%     | 86.84%                 | 54.39%         |
| best KNN acc.      | 53.37%          |       159 |    182 | 91.77% | 92.11%     | 91.23%                 | 59.65%         |
| best 1NN acc.      | 1.17%           |       337 |      4 | 90.48% | 91.23%     | 91.23%                 | 93.86%         |

![](results/figs/breast+cancer+wisconsin+diagnostic.png)


## breast+cancer+wisconsin+prognostic dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |       116 |      0 | 47.77% | 69.23%     | 33.33%                 | 69.23%         |
| best MeATCube F1   | 70.69%          |        34 |     82 | 79.40% | 82.05%     | 41.03%                 | 38.46%         |
| best MeATCube acc. | 70.69%          |        34 |     82 | 79.40% | 82.05%     | 41.03%                 | 38.46%         |
| best KNN acc.      | 96.55%          |         4 |    112 | 74.68% | 79.49%     | 71.79%                 | 66.67%         |
| best 1NN acc.      | 7.76%           |       107 |      9 | 47.77% | 69.23%     | 33.33%                 | 76.92%         |

![](results/figs/breast+cancer+wisconsin+prognostic.png)


## credit+approval dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |       391 |      0 | 65.78% | 69.47%     | 61.83%                 | 67.94%         |
| best MeATCube F1   | 65.73%          |       134 |    257 | 79.52% | 80.15%     | 54.20%                 | 53.44%         |
| best MeATCube acc. | 65.73%          |       134 |    257 | 79.52% | 80.15%     | 54.20%                 | 53.44%         |
| best KNN acc.      | 98.98%          |         4 |    387 | 73.79% | 74.05%     | 68.70%                 | 75.57%         |
| best 1NN acc.      | 99.23%          |         3 |    388 | 55.23% | 56.49%     | 54.20%                 | 75.57%         |

![](results/figs/credit+approval.png)


## dermatology dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |       214 |      0 | 43.62% | 50.00%     | 59.72%                 | 80.56%         |
| best MeATCube F1   | 88.32%          |        25 |    189 | 95.96% | 97.22%     | 81.94%                 | 59.72%         |
| best MeATCube acc. | 88.32%          |        25 |    189 | 95.96% | 97.22%     | 81.94%                 | 59.72%         |
| best KNN acc.      | 94.39%          |        12 |    202 | 74.26% | 83.33%     | 91.67%                 | 80.56%         |
| best 1NN acc.      | 17.76%          |       176 |     38 | 64.41% | 73.61%     | 63.89%                 | 90.28%         |

![](results/figs/dermatology.png)


## glass+identification dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |       128 |      0 | 17.50% | 20.93%     | 62.79%                 | 76.74%         |
| best MeATCube F1   | 67.19%          |        42 |     86 | 56.07% | 79.07%     | 23.26%                 | 53.49%         |
| best MeATCube acc. | 67.19%          |        42 |     86 | 56.07% | 79.07%     | 23.26%                 | 53.49%         |
| best KNN acc.      | 12.50%          |       112 |     16 | 38.56% | 41.86%     | 67.44%                 | 74.42%         |
| best 1NN acc.      | 9.38%           |       116 |     12 | 37.79% | 39.53%     | 67.44%                 | 76.74%         |

![](results/figs/glass+identification.png)


## haberman+s+survival dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |       183 |      0 | 48.74% | 75.41%     | 26.23%                 | 55.74%         |
| best MeATCube F1   | 89.07%          |        20 |    163 | 84.24% | 86.89%     | 63.93%                 | 59.02%         |
| best MeATCube acc. | 90.16%          |        18 |    165 | 81.46% | 86.89%     | 26.23%                 | 63.93%         |
| best KNN acc.      | 77.05%          |        42 |    141 | 71.89% | 78.69%     | 83.61%                 | 57.38%         |
| best 1NN acc.      | 99.45%          |         1 |    182 | 42.45% | 73.77%     | 73.77%                 | 73.77%         |

![](results/figs/haberman+s+survival.png)


## heart+disease dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |       177 |      0 | 12.37% | 26.67%     | 1.67%                  | 45.00%         |
| best MeATCube F1   | 63.28%          |        65 |    112 | 39.69% | 56.67%     | 21.67%                 | 45.00%         |
| best MeATCube acc. | 84.18%          |        28 |    149 | 35.39% | 56.67%     | 21.67%                 | 41.67%         |
| best KNN acc.      | 99.44%          |         1 |    176 | 13.33% | 50.00%     | 50.00%                 | 50.00%         |
| best 1NN acc.      | 42.94%          |       101 |     76 | 28.43% | 45.00%     | 23.33%                 | 51.67%         |

![](results/figs/heart+disease.png)


## hepatitis dataset
|                    | deletion rate   |   cb_size |   step | F1      | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:--------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |        48 |      0 | 46.67%  | 87.50%     | 12.50%                 | 81.25%         |
| best MeATCube F1   | 79.17%          |        10 |     38 | 100.00% | 100.00%    | 12.50%                 | 68.75%         |
| best MeATCube acc. | 79.17%          |        10 |     38 | 100.00% | 100.00%    | 12.50%                 | 68.75%         |
| best KNN acc.      | 97.92%          |         1 |     47 | 11.11%  | 12.50%     | 87.50%                 | 87.50%         |
| best 1NN acc.      | 97.92%          |         1 |     47 | 11.11%  | 12.50%     | 87.50%                 | 87.50%         |

![](results/figs/hepatitis.png)


## ionosphere dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |       210 |      0 | 47.23% | 67.14%     | 64.29%                 | 90.00%         |
| best MeATCube F1   | 67.62%          |        68 |    142 | 96.83% | 97.14%     | 72.86%                 | 85.71%         |
| best MeATCube acc. | 67.62%          |        68 |    142 | 96.83% | 97.14%     | 72.86%                 | 85.71%         |
| best KNN acc.      | 55.24%          |        94 |    116 | 90.82% | 91.43%     | 91.43%                 | 87.14%         |
| best 1NN acc.      | 37.62%          |       131 |     79 | 85.59% | 87.14%     | 70.00%                 | 90.00%         |

![](results/figs/ionosphere.png)


## iris dataset
|                    | deletion rate   |   cb_size |   step | F1      | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:--------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |        90 |      0 | 96.71%  | 96.67%     | 96.67%                 | 90.00%         |
| best MeATCube F1   | 91.11%          |         8 |     82 | 100.00% | 100.00%    | 90.00%                 | 83.33%         |
| best MeATCube acc. | 91.11%          |         8 |     82 | 100.00% | 100.00%    | 90.00%                 | 83.33%         |
| best KNN acc.      | 95.56%          |         4 |     86 | 88.86%  | 90.00%     | 96.67%                 | 73.33%         |
| best 1NN acc.      | 76.67%          |        21 |     69 | 96.71%  | 96.67%     | 90.00%                 | 90.00%         |

![](results/figs/iris.png)


## kaggle+pima+indian+diabetes dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |       460 |      0 | 48.25% | 67.53%     | 34.42%                 | 68.83%         |
| best MeATCube F1   | 68.70%          |       144 |    316 | 87.05% | 88.31%     | 34.42%                 | 54.55%         |
| best MeATCube acc. | 68.70%          |       144 |    316 | 87.05% | 88.31%     | 34.42%                 | 54.55%         |
| best KNN acc.      | 29.57%          |       324 |    136 | 74.55% | 76.62%     | 76.62%                 | 65.58%         |
| best 1NN acc.      | 99.13%          |         4 |    456 | 70.77% | 75.97%     | 34.42%                 | 73.38%         |

![](results/figs/kaggle+pima+indian+diabetes.png)


## kaggle+teaching+assistant+evaluation dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |        90 |      0 | 26.10% | 26.67%     | 30.00%                 | 30.00%         |
| best MeATCube F1   | 50.00%          |        45 |     45 | 69.80% | 70.00%     | 36.67%                 | 53.33%         |
| best MeATCube acc. | 50.00%          |        45 |     45 | 69.80% | 70.00%     | 36.67%                 | 53.33%         |
| best KNN acc.      | 76.67%          |        21 |     69 | 64.53% | 66.67%     | 66.67%                 | 56.67%         |
| best 1NN acc.      | 75.56%          |        22 |     68 | 62.15% | 63.33%     | 60.00%                 | 60.00%         |

![](results/figs/kaggle+teaching+assistant+evaluation.png)


## lenses dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |        14 |      0 | 0.00%  | 0.00%      | 20.00%                 | 0.00%          |
| best MeATCube F1   | 35.71%          |         9 |      5 | 61.11% | 60.00%     | 20.00%                 | 20.00%         |
| best MeATCube acc. | 78.57%          |         3 |     11 | 60.00% | 80.00%     | 60.00%                 | 60.00%         |
| best KNN acc.      | 85.71%          |         2 |     12 | 19.05% | 40.00%     | 60.00%                 | 60.00%         |
| best 1NN acc.      | 85.71%          |         2 |     12 | 19.05% | 40.00%     | 60.00%                 | 60.00%         |

![](results/figs/lenses.png)


## liver+disorders dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |       207 |      0 | 63.57% | 69.57%     | 39.13%                 | 66.67%         |
| best MeATCube F1   | 28.99%          |       147 |     60 | 84.03% | 85.51%     | 39.13%                 | 60.87%         |
| best MeATCube acc. | 28.99%          |       147 |     60 | 84.03% | 85.51%     | 39.13%                 | 60.87%         |
| best KNN acc.      | 84.54%          |        32 |    175 | 75.66% | 76.81%     | 75.36%                 | 56.52%         |
| best 1NN acc.      | 4.83%           |       197 |     10 | 76.27% | 78.26%     | 39.13%                 | 66.67%         |

![](results/figs/liver+disorders.png)


## lung+cancer dataset
|                    | deletion rate   |   cb_size |   step | F1      | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:--------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |        15 |      0 | 72.22%  | 66.67%     | 33.33%                 | 50.00%         |
| best MeATCube F1   | 46.67%          |         8 |      7 | 100.00% | 100.00%    | 16.67%                 | 33.33%         |
| best MeATCube acc. | 46.67%          |         8 |      7 | 100.00% | 100.00%    | 16.67%                 | 33.33%         |
| best KNN acc.      | 86.67%          |         2 |     13 | 11.11%  | 16.67%     | 33.33%                 | 33.33%         |
| best 1NN acc.      | 0.00%           |        15 |      0 | 72.22%  | 66.67%     | 33.33%                 | 50.00%         |

![](results/figs/lung+cancer.png)


## post+operative+patient dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |        51 |      0 | 21.21% | 38.89%     | 50.00%                 | 44.44%         |
| best MeATCube F1   | 80.39%          |        10 |     41 | 39.95% | 55.56%     | 61.11%                 | 61.11%         |
| best MeATCube acc. | 80.39%          |        10 |     41 | 39.95% | 55.56%     | 61.11%                 | 61.11%         |
| best KNN acc.      | 86.27%          |         7 |     44 | 18.52% | 27.78%     | 72.22%                 | 72.22%         |
| best 1NN acc.      | 90.20%          |         5 |     46 | 24.24% | 44.44%     | 66.67%                 | 72.22%         |

![](results/figs/post+operative+patient.png)


## wine dataset
|                    | deletion rate   |   cb_size |   step | F1     | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:-------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |       106 |      0 | 72.54% | 72.22%     | 66.67%                 | 83.33%         |
| best MeATCube F1   | 75.47%          |        26 |     80 | 94.51% | 94.44%     | 52.78%                 | 61.11%         |
| best MeATCube acc. | 75.47%          |        26 |     80 | 94.51% | 94.44%     | 52.78%                 | 61.11%         |
| best KNN acc.      | 95.28%          |         5 |    101 | 85.97% | 86.11%     | 75.00%                 | 69.44%         |
| best 1NN acc.      | 5.66%           |       100 |      6 | 75.21% | 75.00%     | 52.78%                 | 88.89%         |

![](results/figs/wine.png)


## zoo dataset
|                    | deletion rate   |   cb_size |   step | F1      | accuracy   | weighted_nn_accuracy   | 1nn_accuracy   |
|:-------------------|:----------------|----------:|-------:|:--------|:-----------|:-----------------------|:---------------|
| initial            | 0.00%           |        60 |      0 | 73.67%  | 85.00%     | 40.00%                 | 90.00%         |
| best MeATCube F1   | 85.00%          |         9 |     51 | 100.00% | 100.00%    | 50.00%                 | 90.00%         |
| best MeATCube acc. | 85.00%          |         9 |     51 | 100.00% | 100.00%    | 50.00%                 | 90.00%         |
| best KNN acc.      | 91.67%          |         5 |     55 | 32.65%  | 50.00%     | 80.00%                 | 80.00%         |
| best 1NN acc.      | 88.33%          |         7 |     53 | 78.57%  | 80.00%     | 40.00%                 | 90.00%         |

![](results/figs/zoo.png)
