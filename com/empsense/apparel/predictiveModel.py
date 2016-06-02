import pandas
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier

# Read the Apparel data in CSV file.
apperalData = pandas.read_csv("ApperalDataSet.csv")

# Print the names of the columns in CSV dataset.
print(apperalData.columns)

#Feature Extraction
#Select most influenced features from datamart for Churn of an employees based on correlation values
def featureExtraction():
    x = apperalData.corr()["churn"]
    print(x)
    print(x.keys())
    #this should return a list with correct correlation factors

#Create the model
def setModel():
    #get the columns in the csv
    columns = apperalData.columns.tolist()
    # Remove Unwanted Labels to predict labels.But here we have to take the featureExtraction() and do it.
    #have to use only numeric values to the model
    columns = [c for c in columns if
               c not in ["ID", "Name", "Basic Salary", "churn", "Health Status", "Recidency", "Past Job Role",
                         "Education", "Job Role"]]
    # Set the predicted target to Churn
    target = "churn"
    # Generate the training set.  Set random_state to be able to replicate results.
    train = apperalData.sample(frac=0.8, random_state=1)
    # Select anything not in the training set and put it in the testing set.
    test = apperalData.loc[~apperalData.index.isin(train.index)]
    # Print the shapes of both sets.
    print(train.shape)
    print(test.shape)
    return (target,columns,train,test)

#Train the model using different classification methods
#Using Linear Regression
def trainModelLinearRegression(target,columns,train,test):
    # Initialize the model class and calculate linear regression
    model = LinearRegression()
    # Fit the model to the training data.
    model.fit(train[columns], train[target])
    # Generate our predictions for the test set.
    predictions = model.predict(test[columns])
    return predictions

#Using Random Forest
def trainModelRandomForest(target,columns,train,test):
    # Initialize the model with some parameters.
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
    # Fit the model to the data.
    model.fit(train[columns], train[target])
    # Generate our predictions for the test set.
    predictions = model.predict(test[columns])
    return predictions

#Using Desicion Trees
def trainModelDesicionTrees(target, columns, train, test):
    model = DecisionTreeClassifier(random_state=0)
    # Fit the model to the data.
    model.fit(train[columns], train[target].astype(int))
    # Make predictions.
    predictions = model.predict(test[columns])
    return predictions

#Using Support Vector Machine
def trainModelSVM(target, columns, train, test):
    # model = svm.SVC(gamma=0.001, C=100.)
    # model.fit(train[columns], train[target].astype(int))
    # SVC(decision_function_shape=None,random_state=None,verbose=False)
    # # SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    # #   decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
    # #   max_iter=-1, probability=False, random_state=None, shrinking=True,
    # #   tol=0.001, verbose=False)
    # predictions = model.predict(test[columns])
    # print("SVM")
    # print(mean_squared_error(predictions, test[target]))
    return 0

#Test the model using Mean Squared Error
def tetstingModel(predictions,target,test):
    # Compute error between our test predictions and the actual values.
    return mean_squared_error(predictions, test[target])

#Select best model by comparing different error values
def selectBestModel():
    bestModel = "Desicion Trees"
    target,columns,train,test=setModel()

    predictions=trainModelDesicionTrees(target,columns,train,test)
    errorDecesionTree=tetstingModel(predictions, target, test)
    print("errorDecesionTree")
    print(errorDecesionTree)
    predictions =trainModelLinearRegression(target, columns, train, test)
    errorLinearRegression = tetstingModel(predictions, target, test)
    print("errorLinearRegression")
    print(errorLinearRegression)
    predictions =trainModelRandomForest(target, columns, train, test)
    errorRandomForest = tetstingModel(predictions, target, test)
    print("errorRandomForest")
    print(errorRandomForest)
    #errorRandomForest = trainModelSVM(target, columns, train, test)

    #Check this please
    if(errorLinearRegression < errorDecesionTree):
        bestModel="Linear Regression"
    if(errorRandomForest < errorDecesionTree):
        bestModel="Random Forest"
    if (errorRandomForest < errorLinearRegression):
        bestModel = "Random Forest"
    if (errorLinearRegression < errorRandomForest):
        bestModel = "errorLinearRegression"

    print(bestModel)


#Function callings

d=featureExtraction()
selectBestModel()

#have to find a way to feed data to the model to find churn list






