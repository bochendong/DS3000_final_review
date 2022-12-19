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
