Here are some benchmarks that we were able to run so far.

Measurements:

Loss is in "Test/Train"
| Embedding     | Model         | Details  | Significance      | Accuracy (Test/Train) |
| ------------- |:-------------:|----------| -----------------:| --------------------- |
| No            | Random        |          | No                | 0.5019 / 0.5016
| No            | Decision Tree |          | No                | 0.5346 / 0.5376
| No            | XGBoost       |          | No                | 0.5356 / 0.5393
| Yes           | NN KaggleBase | 3 epochs | No                | 0.5348 / 0.5355
| No            | NN KaggleBase |20 epochs | No                | 0.5348 / 0.5344
| No            | NN KaggleBase |20 epochs | Yes               | 0.5341 / 0.5338
| Yes           | NN KaggleBase |20 epochs | No                | 0.5403 / 0.5422
| Yes           | NN KaggleBase |20 epochs | Yes               | 0.5393 / 0.5409
