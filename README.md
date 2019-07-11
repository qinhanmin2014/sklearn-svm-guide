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

- Datasets 4: Breast Cancer (from sklearn.datasets.load_breast_cancer)
  * [jupyter notebook](https://nbviewer.jupyter.org/github/qinhanmin2014/sklearn-svm-guide/blob/master/AX_Breast_Cancer.ipynb)
  *  evaluate using cross validation accuracy
  *  default in libsvm and old default in scikit-learn: 62.74%
  *  new default in scikit=learn: 91.24%
  *  scale with MinMaxScaler: 96.13%
  *  scale with MinMaxScaler & tune the parameters: 97.54%
  *  **scale with StandardScaler: 97.54%**
  *  scale with StandardScaler & tune the parameters: 96.66%

- Datasets 5: Digits (from sklearn.datasets.load_digits)
  * [jupyter notebook](https://nbviewer.jupyter.org/github/qinhanmin2014/sklearn-svm-guide/blob/master/AX_Digits.ipynb)
  *  evaluate using cross validation accuracy
  *  default in libsvm and old default in scikit-learn: 44.88%
  *  new default in scikit=learn: 96.38%
  *  scale with MinMaxScaler: 95.72%
  *  **scale with MinMaxScaler & tune the parameters: 97.33%**
  *  scale with StandardScaler: 94.88%
  *  scale with StandardScaler & tune the parameters: 94.77%

- Datasets 6: Wine (from sklearn.datasets.load_wine)
  * [jupyter notebook](https://nbviewer.jupyter.org/github/qinhanmin2014/sklearn-svm-guide/blob/master/AX_Wine.ipynb)
  *  evaluate using cross validation accuracy
  *  default in libsvm and old default in scikit-learn: 42.77%
  *  new default in scikit=learn: 66.39%
  *  scale with MinMaxScaler: 96.68%
  *  scale with MinMaxScaler & tune the parameters: 96.68%
  *  **scale with StandardScaler: 98.33%**
  *  scale with StandardScaler & tune the parameters: 97.76%

## experiment B: Common Mistakes in Scaling Training and Testing Data
  * [jupyter notebook](https://nbviewer.jupyter.org/github/qinhanmin2014/sklearn-svm-guide/blob/master/B_Common_Mistakes_in_Scaling.ipynb)
  *  evaluate using test set accuracy
  * wrong way: use different scaler for training and testing sets (MinMaxScaler): 69.23% (69.23% in the reference)
  * wrong way: use different scaler for training and testing sets (StandardScaler): 78.21%
  * right way: use same scaler for training and testing sets (MinMaxScaler): 87.50% (89.42% in the reference)
  * **right way: use same scaler for training and testing sets (StandardScaler): 89.42%**

## experiment C: When to Use Linear but not RBF Kernel

- Number of instances << number of features
  * suggestion: use linear kernel
  * [jupyter notebook](https://nbviewer.jupyter.org/github/qinhanmin2014/sklearn-svm-guide/blob/master/C1_Linear_not_RBF_Kernel.ipynb)
  * RBF kernel cross validation accuracy 92.85% (97.22% in the reference)
  * linear kernel cross validation accuracy 92.85% (98.61% in the reference)

- Both numbers of instances and features are large
  * suggestion: use linear kernel
  * [jupyter notebook](https://nbviewer.jupyter.org/github/qinhanmin2014/sklearn-svm-guide/blob/master/C2_Linear_not_RBF_Kernel.ipynb)
  * RBF kernel, cross validation accuracy 97.17% (96.81% in the reference), wall time 15min 17s
  * linear kernel, corss validation accuracy 96.63% (97.01% in the reference), wall time <1s

- Number of instances >> number of features
  * suggestion: if linear kernel, set dual=False (default dual=True)
  * [jupyter notebook](https://nbviewer.jupyter.org/github/qinhanmin2014/sklearn-svm-guide/blob/master/C3_Linear_not_RBF_Kernel.ipynb)
  * dual=False, cross validation accuracy 68.51% (75.67% in the reference), wall time 35s
  * dual=True, corss validation accuracy 68.51% (75.67% in the reference), wall time 10min 31s

## reference

- A Practical Guide to Support Vector Classification, Chih-Wei Hsu et al.

## LIBLINEAR

- https://www.csie.ntu.edu.tw/~cjlin/liblinear/
- https://github.com/cjlin1/liblinear

## LIBSVM

- https://www.csie.ntu.edu.tw/~cjlin/libsvm/
- https://github.com/cjlin1/libsvm
