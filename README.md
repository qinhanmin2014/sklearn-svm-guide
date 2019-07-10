# sklearn-svm-guide
Rapidly obtain acceptable results using SVM (based on scikit-learn)

## common procedure

- Conduct simple scaling on the data
  * sklearn.preprocessing.MinMaxScaler/StandardScaler
- Consider the RBF kernel
  * sklearn.svm.SVC default
- Use cross-validation to find the best parameter C and gamma   
  * sklearn.model_selection.GridSearchCV

## common example

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
X_train, y_train = load_digits(return_X_y=True)
sc = MinMaxScaler(feature_range=(-1, 1))
Xt_train = sc.fit_transform(X_train)
params = {"C": np.logspace(-5, 15, num=11, base=2),
          "gamma": np.logspace(3, -15, num=10, base=2)}
clf = GridSearchCV(SVC(), params, n_jobs=-1)
scores = cross_val_score(clf, Xt_train, y_train)
print(np.mean(scores), "+/-", np.std(scores))
```

## experiment A: Examples of the Proposed Procedure

- Datasets 1: Astroparticle (from the reference)
  * [jupyter notebook](https://nbviewer.jupyter.org/github/qinhanmin2014/sklearn-svm-guide/blob/master/A1_Astroparticle_Physics.ipynb)
  *  evaluate using test set accuracy
  *  default in libsvm and old default in scikit-learn: 66.93% (66.93% in the reference)
  *  new default in scikit=learn: 96.25%
  *  scale with MinMaxScaler: 96.15% (96.15% in the reference)
  *  **scale with MinMaxScaler & tune the parameters: 96.93% (96.87% in the reference)**
  *  scale with StandardScaler: 96.80%
  *  scale with StandardScaler & tune the parameters: 96.68%

- Datasets 2: Bioinformatics (from the reference)
  * [jupyter notebook](https://nbviewer.jupyter.org/github/qinhanmin2014/sklearn-svm-guide/blob/master/A2_Bioinformatics.ipynb)
  *  evaluate using cross validation accuracy
  *  default in libsvm and old default in scikit-learn: 56.53% (56.52% in the reference)
  *  new default in scikit=learn: 81.87%
  *  scale with MinMaxScaler: 78.27% (78.52% in the reference)
  *  **scale with MinMaxScaler & tune the parameters: 84.71% (85.17% in the reference)**
  *  scale with StandardScaler: 56.53%
  *  scale with StandardScaler & tune the parameters: 84.15%

- Datasets 3: Astroparticle (from the reference)
  * [jupyter notebook](https://nbviewer.jupyter.org/github/qinhanmin2014/sklearn-svm-guide/blob/master/A3_Vehicle.ipynb)
  *  evaluate using test set accuracy
  *  default in libsvm and old default in scikit-learn: 2.44% (2.44% in the reference)
  *  new default in scikit=learn: 36.59%
  *  scale with MinMaxScaler: 12.20% (12.20% in the reference)
  *  **scale with MinMaxScaler & tune the parameters: 80.49% (87.80% in the reference)**
  *  scale with StandardScaler: 65.85%
  *  scale with StandardScaler & tune the parameters: 78.05%

- Datasets 4: Bioinformatics (from sklearn.datasets.load_digits)
  * [jupyter notebook](https://nbviewer.jupyter.org/github/qinhanmin2014/sklearn-svm-guide/blob/master/AX_Digits.ipynb)
  *  evaluate using cross validation accuracy
  *  default in libsvm and old default in scikit-learn: 44.88%
  *  new default in scikit=learn: 96.38%
  *  scale with MinMaxScaler: 95.72%
  *  **scale with MinMaxScaler & tune the parameters: 97.33%**
  *  scale with StandardScaler: 94.88%
  *  scale with StandardScaler & tune the parameters: 94.77%


## reference

- A Practical Guide to Support Vector Classification, Chih-Wei Hsu et al.
- https://www.csie.ntu.edu.tw/~cjlin/libsvm/
