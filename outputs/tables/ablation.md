| variant                        | model               |   n_features |   ROC-AUC all |   PR-AUC all |   ROC-AUC active |   PR-AUC active |
|:-------------------------------|:--------------------|-------------:|--------------:|-------------:|-----------------:|----------------:|
| 1. demographics only           | logistic regression |            8 |         0.615 |        0.429 |            0.508 |           0.095 |
| 1. demographics only           | LightGBM            |            8 |         0.775 |        0.671 |            0.533 |           0.101 |
| 2. + behavioural               | logistic regression |           22 |         0.986 |        0.980 |            0.932 |           0.687 |
| 2. + behavioural               | LightGBM            |           22 |         0.985 |        0.980 |            0.928 |           0.704 |
| 3. + engagement                | logistic regression |           32 |         0.991 |        0.987 |            0.955 |           0.812 |
| 3. + engagement                | LightGBM            |           32 |         0.990 |        0.987 |            0.954 |           0.799 |
| 4. + support (full)            | logistic regression |           36 |         0.992 |        0.989 |            0.962 |           0.828 |
| 4. + support (full)            | LightGBM            |           36 |         0.992 |        0.988 |            0.960 |           0.823 |
| demographics + engagement only | logistic regression |           18 |         0.984 |        0.979 |            0.925 |           0.711 |
| demographics + engagement only | LightGBM            |           18 |         0.984 |        0.978 |            0.922 |           0.707 |
| demographics + support only    | logistic regression |           12 |         0.602 |        0.420 |            0.597 |           0.139 |
| demographics + support only    | LightGBM            |           12 |         0.795 |        0.686 |            0.580 |           0.172 |
| full - behavioural             | logistic regression |           22 |         0.987 |        0.982 |            0.937 |           0.754 |
| full - behavioural             | LightGBM            |           22 |         0.986 |        0.982 |            0.933 |           0.756 |