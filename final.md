- [Linear Model](#linear-model)
  - [LAD and RSS](#lad-and-rss)
  - [sklearn Linear Regression](#sklearn-linear-regression)
  - [Multi Varibale Linear Regression](#multi-varibale-linear-regression)
- [R2 score](#r2-score)
- [Likelihood](#likelihood)
- [Class Balanced](#class-balanced)
- [Logistic Regresssion](#logistic-regresssion)
  - [Bernoulli model](#bernoulli-model)
  - [Log likelihood](#log-likelihood)
  - [Gradient for logistic regression](#gradient-for-logistic-regression)
  - [sklearn Logistic regression](#sklearn-logistic-regression)
- [Performance](#performance)
  - [Measurement:](#measurement)
- [Threshold](#threshold)
- [AUC and ROC](#auc-and-roc)
- [Confidence Interval](#confidence-interval)
  - [Using central limit theorem to compute confidence interval](#using-central-limit-theorem-to-compute-confidence-interval)
  - [Using t-distribution to compute confidence interval](#using-t-distribution-to-compute-confidence-interval)
- [Bootstrap](#bootstrap)
- [Trees](#trees)
- [Dimension Reduction](#dimension-reduction)
- [Clustering](#clustering)

# Linear Model

L1 loss function (sum of magnitudes, used for LAD model):

$$L_1(\theta) = \sum_{i=1}^{n} \lvert {y_i-\hat{y_i}} \rvert$$

L2 loss function (RSS, residual sum of squares, used for OLS model):

$$L_2(\theta) = \sum_{i=1}^{n} ({y_i-\hat{y_i}})^2$$

## LAD and RSS

```Python
def linearModelPredict(b,X):
    yp = X @ b
    return yp

def linearModelLossLAD(b,X,y):
    y_pred = linearModelPredict(b, X)
    residual = y - y_pred
    sum_abs_dev = sum(abs(residual))
    grad = -np.dot(np.sign(residual), X)
    return (sum_abs_dev, grad)

def linearModelLossRSS(b,X,y):
    y_pred = linearModelPredict(b,X)
    residual = y - y_pred
    residual_sum_of_squares = sum(residual**2)
    #compute the gradient
    gradient = -2 * np.dot(np.transpose(X),residual)
    return (residual_sum_of_squares, gradient)

def linearModelFit(X,y,lossfcn = linearModelLossRSS):
    nrows, ncols = X.shape
    b = np.zeros((ncols, 1))

    RESULT = so.minimize(lossfcn, b, args=(X, y),jac=True)
    estimated_betas = RESULT.x

    TSS = sum((y - np.mean(y))**2)
    RSS, deriv = linearModelLossRSS(estimated_betas, X, y)

    R2 = 1-(RSS / TSS)
    return (estimated_betas, R2)
```

```Python
possum_data = pd.read_csv("./possum.csv")
x = possum_data['age']

X = np.c_[np.ones(len(possum_data)), x.to_numpy()]
y = possum_data["tailL"].to_numpy()

betas_RSS, r2_rss = linearModelFit(X,y)
betas_LAD, r2_LAD = linearModelFit(X,y, linearModelLossLAD)

y_pred_RSS = linearModelPredict(betas_RSS, X)
y_pred_LAD = linearModelPredict(betas_LAD, X)
plt.plot(X[:,1], y_pred_RSS, color='red', linestyle='--')
plt.plot(X[:,1], y_pred_LAD, color='blue', linestyle='--')
plt.scatter(x, y, alpha= 0.8)
plt.legend(['RSS', 'LAD'])
plt.plot()

```

## sklearn Linear Regression

```Python
data = pd.read_csv("Real_estate_valuation_dataset.csv")
longitude = data['longitude']
latitude = data['latitude']

x = np.c_[np.ones(len(data)), data[['longitude', 'latitude']].to_numpy()]
y = data['house_price_of_unit_area'].to_numpy()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=seed)
```

```Python
model = LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
r2_test = r2_score(ytest, ypred)
print("Coefficient of determination on the test set: %.3f" % r2_test)

# Obtain intercept and coefficient
intercept = model.intercept_
coefficient = model.coef_
print("Intercept is:", intercept.round(3))
print("Coefficient are:", coefficient.round(3))
```

## Multi Varibale Linear Regression

```Python
x = data.drop('house_price_of_unit_area', axis=1)
x = np.c_[np.ones(len(data)), x.to_numpy()]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
```

## Ridge Regression
```Python
# Define the model
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)

# Fit the model using the training set
ridge_reg.fit(x, y)

# Obtain intercept and coefficient
intercept = ridge_reg.intercept_
coefficient = ridge_reg.coef_

#prediction
ridge_reg.predict([[1.5]])

coefficient_feature = coefficient[1:]
features_name = data.columns[:-1]
plt.barh(features_name, coefficient_feature)
plt.xlabel('Coefficient')
```

# R2 score

```Python
def cal_r2(y, ypred):
    RSS = np.sum((y - ypred) ** 2)
    TSS = sum((y - np.mean(y))**2)
    return 1 - RSS/TSS
```

# Likelihood

The negative log likelihood for a Poisson random variable is

$$\ell(\lambda; \mathbf{y}) = -\sum_{i=1}^N\Bigg(   y_{i}\cdot \ln(\lambda) - \lambda - \ln(y_i!) \Bigg)$$

Write a function called `modelPrediction` which accepts as its first argument a vector of coefficents $\beta$ and a design matrix $\mathbf{X}$. The function should return predictions of the form $\widehat{\mathbf{y}} = \exp(\mathbf{X}\beta)$. Then write a function called `fitModel` which accepts as its first argument argument a design matrix $\mathbf{X}$ and as its second argument a vector of outcomes counts $\mathbf{y}$. The function should return the maximum likelihood estimates for the coefficients of a Poisson regression of $\mathbf{y}$ onto $\mathbf{X}$.

```Python
def poissonNegLogLikelihood(lam,y):

    # Read up on the gamma function to make sure you get the likelihood right!
    neg_log_lik = - np.sum(y * np.log(lam) - lam)
    return neg_log_lik

def poissonRegressionNegLogLikelihood(b, X, y):
    #Enter the expression for lambda as shown above!
    lam = np.exp(np.dot(X, b))
    # Use poissonNegLogLikelihood to compute the likelihood
    neg_log_lik = poissonNegLogLikelihood(lam,y)
    return neg_log_lik
```

```Python
def modelPrediction(b,X):
    yhat = np.exp(np.dot(X, b))
    return yhat

def fitModel(X,y):
    # Instantiate a guess for the betas, beta_start, so that the optimizer has somewhere to start
    # Keep in mind what shape the beta_start should be. It should have the same number of elements as X as columns
    beta_start = np.zeros(X.shape[1])
    # Minimize the appropriate likelihood function
    mle = minimize(poissonRegressionNegLogLikelihood, beta_start, args=(X,y), method="Powell", tol=1e-8)
    # Extract the maximum likelihood estimates from the optimizer.
    betas = mle.x
    return betas
```

```Python
df = pd.read_csv("poisson_regression_data.csv")
x = df.x.values
X = np.c_[np.ones(x.size), x]
y= df.y.values

betas = fitModel(X, y)

# Make predictions on new data
newx = np.linspace(-2,2,1001)
newX = np.c_[np.ones(newx.size), newx]
y_predicted = modelPrediction (betas, newX)

fig, ax = plt.subplots(dpi = 120)
ax.scatter(x, y)
ax.plot(newx, y_predicted)
plt.show()
```

# Class Balanced

This can be used for a baselien accuracy test.

```Python
df = pd.read_csv('diabetes.csv')
counts= df.Outcome.value_counts()
# Counts have a great difference(>1:2) means inbalanced.
# In linear regression a label with > 5% of instances is statistically significant.

baseline_accuracy = round(counts[0]/(counts[0]+counts[1]), 3)
print("Baseline Accuracy is:", baseline_accuracy)
```

# Logistic Regresssion

## Bernoulli model

Logistic regression are come from the Bernoulli model, where we assume X is a random varibale, then:

$$
Pr(X= 1) = p = 1 - Pr(X = 0) = 1 - q
$$

The probability mass function $f$ of this distribution, over possible outcomes k, is

$$
f(x;p) =

\left \{
\begin{aligned}
    &p & \text{if } x = 1\\
    &1 - p& \text{if } x = 0
\end{aligned}
\right.
$$

This can also be expressed as:

$$
f(x;p) = p^k(1-p)^{1 - k}
$$

Let us consider the binary classification problem with labels $y\in \{\pm 1\}$. With the parameterization:

$$
Pr(Y = 1| X= x) =:p(x;w)
$$

where p is a function that maps x and w into $[0,1]$, we then have the Bernoulli model for generating the label $y \in \{\pm 1\}$:

$$
Pr(Y = y| X= x) = p(x; w)^{y}(1 - p(x; w))^{1 - y}
$$

This can also be expressed as:

$$
Pr(Y = y| X= x) =

\left \{
\begin{aligned}
    &p(x; w) & \text{if } Y = 1\\
    &1 - p(x; w)& \text{if } Y = 0
\end{aligned}
\right.
$$

## Log likelihood

Then if we consider $\mathcal{D} = \{(x_i, y_i): i = 1, \cdots, n\}$ as an i.i.d sample, we can write the condition liklihood:

$$
\begin{aligned}
&Pr(Y_1 = y_1, \cdots, Y_n = y_n | X_1 = x_1, \cdots, X_n = x_n) \\
&= \prod_{i = 1}^n Pr(Y_i = y_i| X_i = x_i)\\
&= \prod_{i = 1}^n p(x_i; w)^{y_i}(1 - p(x_i; w))^{1- y_i}
\end{aligned}
$$

One natural chioce of $p(x;w)$ is:

$$
p(x;w) = w^Tx + b
$$

However, this choice has the disadvantage in the sense that the left-hand side takes value in $[0, 1]$ while the right-hand side takes value in $\mathbb{R}$, so we first take a logit transformation of $p$ and then equate it to an affine function:

$$
\begin{aligned}
\log \frac{p(x;w)}{1 - p(x;w)} &= w^Tx +b\\
p(x; w) &= \frac{1}{1 - e^{-(w^Tx + b)}}\\
p(x; w) &= \sigma(w^Tx +b)
\end{aligned}
$$

Take log we get:

$$
\begin{aligned}
\log(L(\textbf{w}))&= \log \prod_{i = 1}^n \sigma(\textbf{w}^Tx)^{y_i}(1 - \sigma(\textbf{w}^Tx))^{1- y_i}\\
&= \sum_{i = 1}^n y_i \log(\sigma(\textbf{w}^Tx)) + (1 - y_i)\log(1 - \sigma(\textbf{w}^Tx))

\end{aligned}
$$

## Gradient for logistic regression

First, we have the sigmod function:

$$
\sigma(w^Tx) = \frac{1}{1 + e^{-w^Tx}}
$$

Then we take the derivate with respect to w, we have:

$$

\begin{aligned}
\frac{\partial \sigma(w^Tx)}{\partial  w} &= \frac{e^{-w^Tx}}{( 1+ e^{-w^Tx})^2}x\\
\frac{ \partial (1 - \sigma(w^Tx))} {\partial w} &= - \frac{e^{-w^Tx}}{( 1+ e^{-w^Tx})^2}x

\end{aligned}
$$

Next we consider the derivate of $\log(L(w))$ with respect to w, we have:

$$
\begin{aligned}
    \frac{\partial \log(L(w))}{\partial w} &= \sum_{i=1}^n \frac{\partial y_i \log(\sigma(w^Tx))}{\partial w} + \frac{\partial (1 - y_i) \log(1 - \sigma(w^Tx))}{\partial w}\\

    &= \sum_{i=1}^n y_i
    \frac{ \partial \log(\sigma(w^Tx))}{\partial \sigma(w^Tx)} \frac{\partial \sigma(w^Tx)}{\partial w} +
    (1 - y_i)  \frac{ \partial \log(1 - \sigma(w^Tx))} {\partial (1 - \sigma(w^Tx))}\frac{ \partial (1 - \sigma(w^Tx))} {\partial w}

    \\

    &=  \sum_{i=1}^n y_i (1 + e^{-w^Tx})\frac{e^{-w^Tx}}{( 1+ e^{-w^Tx})^2}x
    -
     (1 - y_i)\frac{1 + e^{-w^Tx}}{e^{-w^Tx}} \frac{e^{-w^Tx}}{( 1+ e^{-w^Tx})^2}x
    \\
    &= \sum_{i=1}^n y_i \frac{e^{-w^Tx}}{ 1+ e^{-w^Tx}}x - (1 - y_i)\frac{1}{1 + e^{-w^T}}x\\

    &= \sum_{i=1}^n y_i \frac{e^{-w^Tx}}{ 1+ e^{-w^Tx}}x+ y_i\frac{1}{1 + e^{-w^T}}x - \frac{1}{1 + e^{-w^T}}x\\

    & = \sum_{i=1}^n ( y_i \frac{e^{-w^Tx}}{ 1+ e^{-w^Tx}} +  y_i\frac{1}{1 + e^{-w^T}} - \frac{1}{1 + e^{-w^T}}) x\\

    &= \sum_{i=1}^n (y_i - \frac{1}{1 + e^{-w^T}})x\\

    &= \sum_{i=1}^n (y_i - \sigma(w^Tx))x
\end{aligned}
$$

And we let the loss function be $J(w)$, where

$$
\begin{aligned}
J(w) &= - \log(L(w))\\
\nabla J(w) &= \sum_{i=1}^n ( \sigma(w^Tx_i) - y_i)x_i

\end{aligned}
$$

## sklearn Logistic regression

```Python
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression(penalty='none', max_iter=10000)
# penalty: "l1"(Lasso), "l2"(Ridge)(default), "none", "elasticnet"(l1+l2)
# small coefficients if a penalty is applied.
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

ytest_prob = model.predict_proba(Xtest)
```

# Performance

- **TP = “True Positive”.** i.e. your classifier predicted 1; it was correct.
- **FP = “False Positive”.** i.e. your classifier predicted 1; it was incorrect (Type I error).
- **FN = “False Negative”.** i.e. your classifier predicted 0; it was incorrect (Type II error).
- **TN = “True Negative”.** i.e. your classifier predicted 0; it was correct.

## Measurement:

- Precision: What percentage of all positive predictions were correct?
  $$
  Precision = \frac{\# \ \text{True Positive}}{\#\ \text{Predicted Positive}}
  $$
- Recall: What percentage of all positive samples were recalled?

$$
Recall = \frac{\# \ \text{True Positive}}{\# \ \text{Class Positive}}
$$

- F-Measure: A combined metric which accounts for precision and recall in a single measure.

$$
F-Measure = 2 * (\frac{\text{Precision} * \text{Recall}}{\text{Precision} + {Recall}})
$$

```Python
def compute_performance(ypred, ytest, classes):
    tp = sum(np.logical_and(ypred == classes[1], ytest == classes[1]))
    tn = sum(np.logical_and(ypred == classes[0], ytest == classes[0]))
    fp = sum(np.logical_and(ypred == classes[1], ytest == classes[0]))
    fn = sum(np.logical_and(ypred == classes[0], ytest == classes[1]))

    return tp, tn, fp, fn

tp, tn, fp, fn = compute_performance(ypred, ytest, model.classes_)
Acc = (tp + tn) / (tp + tn + fp + fn)
print("Acc: %.5f" % Acc)
```

# Threshold

```Python
threshold = 0.6

ytest_prob = model.predict_proba(Xtest)
ypred = model.classes_[(ytest_prob[:,1]>threshold).astype(int)]
tp, tn, fp, fn = compute_performance(ypred, ytest, model.classes_)
Acc = (tp + tn) / (tp + tn + fp + fn)
print("Acc: %.5f" % Acc)
```

# AUC and ROC

- If Area Under Receiver Operating
  Characteristic Curve (AUROC) is
  1, classifier is perfect
- If AUROC is 0.5, classifier is no
  better than random chance.
- If AUROC is 0, classifier is
  worst possible (TPR=0, FPR=1)

```Python
fpr, tpr, _ = roc_curve(ytest, ytest_prob[:,1], pos_label=1)
ax = sns.lineplot(x = fpr, y = tpr)

AUC = auc(fpr, tpr)
print("The AUC is : "+ str(AUC))
```

# Confidence Interval

## Using central limit theorem to compute confidence interval

```Python
# Standard Error
stderr = np.std(data.Match, ddof=1) / np.sqrt(len(data.Match))

# alpha = 0.05
critval = 1.96 # critical value

# Confidence interval
norm_ci = [((data.Match.mean() - critval*stderr)*100).round(1),((data.Match.mean() + critval*stderr)*100).round(1)]
```

## Using t-distribution to compute confidence interval

The $100(1-\alpha)\%$ confidence interval is

$$ \bar{x} \pm t\_{1-\alpha/2, n-1} \dfrac{\hat{\sigma}}{\sqrt{n}} $$

```python
alpha = (5/100)/2
dof = n-1
crit_val = 1-alpha

t_quantile = t.ppf(crit_val, df=dof)
# 1st arg: critical value; 2nd arg: dof

t_ci = data.Match.mean() + t_quantile * stderr * np.array([-1, 1])
```

# Bootstrap

# Cross Validation

## One hot encoding VS Label encoding
Both can turn category into numeric variables
In most scenarios, one hot encoding is the preferred way to convert a categorical variable into a numeric variable because label encoding makes it seem that there is a ranking between values.

```Python
# One Hot Encoding
model_data = data.copy()
dataOHE = pd.get_dummies(dataOHE, drop_first=True) 
# the dropped one is collinear to the other two

# Label Encoding
label_encoder = preprocessing.LabelEncoder()
dataLE['work_rate_att']= label_encoder.fit_transform(dataLE['work_rate_att'])
print(dataLE.loc[dataLE['work_rate_att'] == 0].work_rate_att.count())
print(dataLE.loc[dataLE['work_rate_att'] == 1].work_rate_att.count())
print(dataLE.loc[dataLE['work_rate_att'] == 2].work_rate_att.count())
```

```Python
# Define Kfold crossvalidation 
x = np.arange(20)
print(x,'\n')
kf = KFold(n_splits=5, shuffle=False)
# Shuffle = True : random test set
for train, test in kf.split(x):
    print("Train set: %s, Test set: %s" % (train, test))

# Leave One Out crossvalidation when dataset is too small
x=np.arange(20)
loo = LeaveOneOut()
for train,test in loo.split(x):
    print("%s %s" % (train, test))
```
```Python
# KFold cross-validated loss without shuffling
kf = KFold(n_splits=5, shuffle=False, random_state=None)
cv_scores = cross_val_score(LinearRegression(), Xtrain, ytrain, cv=kf, scoring=sc)

# KFold cross-validated loss with shuffling
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
cv_scores = cross_val_score(LinearRegression(), Xtrain, ytrain, cv=kf, scoring=sc)

# Leave One Out cross-validated loss
kf = LeaveOneOut()
cv_scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring=sc) 
# for Leave One Out you use the full data set

print(f'List of CV loss:', cv_scores)
print(f"Average CV loss: %.3f +/- %.3f" % (cv_scores.mean(), cv_scores.std()))
```
Model Choose  
Choose the simplest model which has the lowest cross-validated training loss.

# Trees

# Dimension Reduction

# Clustering
