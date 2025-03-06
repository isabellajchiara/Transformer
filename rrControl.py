random.seed(126)
data = pd.read_csv("fullDatasetSY.csv")
data = data.sample(frac=1)

X = data.drop(['Unnamed: 0','0','1','2','3'],axis = 1)
threshold = 0.01
X = X.drop(X.var()[X.var() < threshold].index.values, axis=1)

y = data["3"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.33, shuffle=True)
xTest, xValid, yTest, yValid = train_test_split(xTest, yTest, test_size=0.5, shuffle=True)

'''determine best alpha value for RR'''

alphaValues = {}
params =[0.01,0.1,1,10,100,1000]
for values in params:
  LR = linear_model.Ridge(values)
  LR.fit(xTrain, yTrain)
  yPred = LR.predict(xValid)
  yValid = np.array(yValid)
  alphaValues[values] = np.corrcoef(yPred, yValid)[0,1]
  alphaVal = max(alphaValues, key=alphaValues.get)

print(alphaVal)


X = data.drop(['Unnamed: 0','0','1','2','3'],axis = 1)
threshold = 0.01
X = X.drop(X.var()[X.var() < threshold].index.values, axis=1)

y = data["3"]

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.33, shuffle=True)
xTest, xValid, yTest, yValid = train_test_split(xTest, yTest, test_size=0.5, shuffle=True)
# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store scores
rmse_scores = []
correlation_scores = []

foldAccs = []

# Perform cross-validation
for train_index, test_index in kf.split(X):
    # Split the data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model
    RR = linear_model.Ridge(alphaVal)
    RR.fit(X_train, y_train)

    # Make predictions
    y_pred = RR.predict(X_test)

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_scores.append(rmse)

    # Compute Pearson correlation
    corr, _ = pearsonr(y_test, y_pred)
    correlation_scores.append(corr)
    foldMean = np.mean(correlation_scores)
    foldAccs.append(foldMean)

print(np.mean(foldAccs))
