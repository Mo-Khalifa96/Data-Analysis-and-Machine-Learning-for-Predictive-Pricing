#DATA ANALYSIS AND MACHINE LEARNING FOR PREDICTVE PRICING (PREDICTING CAR PRICES)


#Importing modules to use 
import re 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression, Lasso 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score 
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import warnings
warnings.simplefilter("ignore")
#Adjusting data display options 
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



#Part One: Reading and Inspecting Data 
#1. Loading and reading excel file 
#Loading the dataset onto a dataframe
df = pd.read_excel('Automobile Dataset.xlsx')

#Previewing the first 5 entries 
print(df.head())
print('')


#2. Inspecting the data 
#Inspecting the shape of the dataframe 
shape = df.shape 
print('Number of coloumns:', shape[1])
print('Number of rows:', shape[0]) 
print('')


#Inspecting the coloumn headers, data type, and number of entries 
print(df.info())
print('')



#Part Two: Updating and Cleaning Data 
#1. Identifying and handling missing values 
#We can see in the dataset some entries containing the special character '?', instead of 
# real values. I'll identify and remove or replace them by the appropriate values

#First, reporting coloumns with inappropriate entries and their total count
print('Number of inappropriate entries per coloumn:')
for col in df.columns: 
    if any(df[col].astype('str').str.contains('\?')):
        print(f'{col}:', df[col].astype('str').str.contains('\?').sum())

print('')


#Now replacing the innappropriate values (i.e., '?') with NaN (Not-a-Number) values 
# before dealing with them in the most optimal way for a given coloumn
df.replace('?', np.nan, inplace=True)

#Checking the the number of NaN values for each coloumn
print('Number of null/NaN values per coloumn:')
for col in df.columns:
    if df[col].isna().sum() > 0:
        print(f'{col}:', df[col].isna().sum())

print('')


#Dealing with missing values 
#Dropping rows with missing prices
#check number of rows before 
print('Number of rows before removal:', len(df))

#Drop rows and resetting the index
df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

#Check number of rows after
print('Number of rows after removal:', len(df))
print('')


#Replacing missing values by mode 
#For the missing values in the 'num-of-doors' coloumn, I will replace them by the mode 
# value ('four') since it is the most frequent and thus most likely to occur 
mode_val = df['num-of-doors'].value_counts().idxmax()
df['num-of-doors'].replace(np.nan, mode_val, inplace=True)


#Replacing missing values by mean 
#For the rest of the coloumns with missing values, I will replace the values with the mean 
# value for a given coloumn
#Iterating over each coloumn and updating its missing values  
for col in  df.columns:
    if df[col].isna().sum() > 0:
        #for a given coloumn, get the mean value 
        mean_val = df[col].astype('float64').mean()
        #replace NaN values with the mean 
        df[col].replace(np.nan, mean_val, inplace=True)


#Rechecking the number of missing entries in the dataset again 
print('Number of null/NaN values per coloumn:')
print(df.isna().sum())
print('')


#Now the dataset is all cleaned up and free of any missing or null entries. Next I will assign 
# the correct data types to the coloumns that require correcting.


#2. Correcting data format 
#As seen earlier, some of the coloumns are assigned the wrong data type, a problem that would hinder the analysis. 
# I will identify such coloumns and correct their data format.

#Previewing the dataframe again 
print(df.head())
print('')

#We can check again each coloumn and identify the ones with incorrect data types
print('Coloumns with incorrect data types:')
for col in df.columns: 
    #return coloumn name only if it's convertable to number but is assigned the wrong data type 
    if df[col].dtype == 'object':
        try: 
            df[col].astype('float')
            print(' ', col)
        except:
            continue 

print('')


#Coverting data types to proper format
#Converting data from object to integer
df[['normalized-losses', 'horsepower', 'peak-rpm', 'price']] = df[['normalized-losses', 'horsepower', 'peak-rpm', 'price']].astype('int64')

#Converting data from object to float-point number
df[['bore', 'stroke']] = df[['bore', 'stroke']].astype('float64')


#We can check the coloumns data types again 
print(df.dtypes)
print('')


#3. Dealing with categorical variables (One Hot Encoding)
#To render categorical variable viable for numerical and statistical analysis, we have first to transform them 
# to numerical variables. To do so, I will use one hot encoding. 

#First, identifying the coloumns with categorical variables 
categorical_cols = []
for col in df.columns: 
    if df[col].dtype == 'object': 
        print(col)
        categorical_cols.append(col)
print('')


#Now performing one hot encoding on these coloumns 
#Get encoder object 
encoder = OneHotEncoder(handle_unknown='ignore')

#Perform one hot encoding and assign feature names 
df_encoded_vars = pd.DataFrame(encoder.fit_transform(df[categorical_cols]).toarray())
df_encoded_vars.columns = encoder.get_feature_names(categorical_cols)

#Get new dataframe with the new encoded categories 
df_new = df.join(df_encoded_vars)

#Previewing the new dataframe 
print(df_new.head())
print('\n\n')



#Part Three: Data Selection and Preprocessing
#In this section, I will identify the subset of data that will be used for developing the model and making the necessary 
# preperations and adjustments to optimize the model's capacity for predictive pricing.


#1. Identifying the variables
#First, we need to identify the car attributes that are most relevant for predicting the final car prices. These will be 
# the features based on which the model will generate its price predictions. To do so, I will apply correlational analysis
# for the numerical variables and One Way ANOVA testing for the categorical ones. The variables with the strongest association
# to the target, price, will be selected for model development.

#1.1 Correlational analysis for numerical variables
#Performing pearson correlation on the numerical variables 
correlations_ByPrice = df.corr()['price'].sort_values(ascending=False)
print(correlations_ByPrice)
print('')

#Visualizing correlations using a heatmap 
plt.figure(figsize=(12,8))     
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), annot=True, mask=mask, cmap='Blues', vmin=-1, vmax=1)
plt.show()
print('')


#1.2 One Way Anova for categorical variables
#Performing One Way ANOVA on the categorical variables 
sig_ByVariable = {}
for col in df.columns:
    if df[col].dtype == 'object':
        if "-" in str(col):
            col_renamed = '_'.join(col.split('-'))
            df_copy = df.copy()
            df_copy.rename(columns={f'{col}': f'{col_renamed}'}, inplace=True)
            anova_model = ols(f'price ~ C({col_renamed})', data=df_copy).fit()
            anova_table = sm.stats.anova_lm(anova_model, typ=2)
            print(anova_table, '\n')
        else:
            anova_model = ols(f'price ~ C({col})', data=df).fit()
            anova_table = sm.stats.anova_lm(anova_model, typ=2)
            print(anova_table, '\n')

        sig_ByVariable[f'{col}'] = anova_table['PR(>F)'].values[0]

print('')

#Now, sorting the ANOVA results by most significant to least significant 
sig_ByVariable_sorted = sorted([(val, key) for (key, val) in sig_ByVariable.items()])    
for val, key in sig_ByVariable_sorted:
    print('{}: {:,.3f}'.format(key, val))

print('')


#Updating the dataframe
#Identifying and removing the unnecessary coloumns 
unnecessary_cols = (categorical_cols + [col for col in df_new.columns if re.findall('fuel-type|num-of-doors', col)]
                    + [col for col in correlations_ByPrice.index if abs(correlations_ByPrice.loc[col]) < 0.5])


#removing unnecessary coloumns and obtaining a new, updated dataframe 
df_updated = df_new.drop(unnecessary_cols, axis=1)

#Previewing the final dataframe 
print(df_updated.head())
print('')


#Selecting the predictor and target variables
#Now selecting the variables for training the model. We will have a total of 17 predictors or independent variables, the car 
# attributes proved to be most relevant, and 1 targer or dependent variable, price.

#specifying the predictor variables and assigning them to 'x_data'
x_data = df_updated.drop('price', axis=1)

#specifying the target variable, price, and assigning it to 'y_data'
y_data = df_updated['price']


#2. Data Splitting 
#Performing data splitting to obtain a training set (80%) and testing set (20%)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.80, random_state=0)

#Check the size of both sets 
print('Number of training samples:', x_train.shape[0])    
print('Number of testing samples:', x_test.shape[0])
print('')


#3. Feature Scaling: Normalizing the Scales 
#Normalizing the scales so that the data would exhibit the same distribution from 0 to 1.

#First, identify the numerical variables and assign them to 'num_vars' 
num_vars = [col for col in correlations_ByPrice.index if 1 > abs(correlations_ByPrice.loc[col]) > 0.5]

#Now normalizing only those variables 
#get a scaler object 
Scaler = MinMaxScaler()

#fitting and scaling the training set 
num_train_scaled = Scaler.fit_transform(x_train[num_vars]) 

#scaling the testing set 
num_test_scaled = Scaler.transform(x_test[num_vars])


#replacing unscaled coloumns with their scaled counterparts 
x_train.drop(num_vars, axis=1, inplace=True)
x_test.drop(num_vars, axis=1, inplace=True)
x_train, x_test = np.concatenate([num_train_scaled, x_train], axis=1), np.concatenate([num_test_scaled, x_test], axis=1)



#Part Four: Model Development and Evaluation

#MODEL ONE: MULTIPLE LINEAR REGRESSION MODEL 
#A multiple regression model is a type of regression model that depicts a linear relationship between multiple predictors and the target. In 
# this case the model will capture the relationship between the relevant car attribute selected earlier and price. 

#Create a regression object 
multireg_model = LinearRegression()

#Fitting the model using the training data 
multireg_model.fit(x_train, y_train)

#Evaluating the model with the testing set using the R-Squared metric
R2_test = multireg_model.score(x_test, y_test)
print(f'The R-squared score for the multiple regression model is: r2={round(R2_test,3)}')
print('')


#Now we can use the first model to generate price predictions and compare them to the actual prices 
#Generating predictions using the testing set
Y_pred = multireg_model.predict(x_test) 

#We can compare the actual prices vs. predicted prices 
Actual_vs_Predicted = pd.concat([pd.Series(y_test.values), pd.Series(Y_pred)], axis=1, ignore_index=True).rename(columns={0:'Actual Prices', 1:'Predicted Prices'})
Actual_vs_Predicted['Actual Prices'] = Actual_vs_Predicted['Actual Prices'].apply(lambda price: '${:,.2f}'.format(price))
Actual_vs_Predicted['Predicted Prices'] = Actual_vs_Predicted['Predicted Prices'].apply(lambda price: '${:,.2f}'.format(price))

#Previewing the first 10 price comparisons 
print(Actual_vs_Predicted.head(10))
print('')


#Model Evaluation: Root Mean Squared Error
#Calculate the mean squared error (MSE)
MSE = mean_squared_error(y_test, Y_pred)

#Get square root of MSE to obtain root mean squared error (RMSE)
RMSE = np.sqrt(MSE)
print(f'The root mean squared error is: RMSE={round(RMSE,3)}')
print('')


#Model Evaluation: Distribution Plot
#Visualizing the distribution of actual vs. predicted prices 
#Creating the distribution plot 
ax1 = sns.distplot(y_test, hist=False, label='Actual Values')
sns.distplot(Y_pred, ax=ax1, hist=False, label='Predicted Values')
#Adding a title and labeling the axes
plt.title('Actual vs. Predicted Values for Car Prices\n(Multiple Regression Model)')
plt.xlabel('Car Price (in USD)', fontsize=12)
plt.ylabel('Distribution density of price values', fontsize=12)
plt.legend(loc='best')
#Adjusting the x-axis to display the prices in a reader-friendly format
plt.gcf().axes[0].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))
plt.xticks(rotation=90)

#Displaying the distribution plot
plt.show()



#MODEL TWO: Multivariate Polynomial Regression Model 
#Performing cross validation to identify the best polynomial degree for the model
#First, specifying the polynomial degrees to test out
poly_orders = [2,3,4]       #up to four polynomials 

#Now looping through the different polynomials and using cross validation determine the most optimal one 
cv_scores = {}
for order in poly_orders: 
    #creating polynomial features object
    poly_features = PolynomialFeatures(degree=order)
    #transforming predictor variables to polynomial features
    x_train_poly = poly_features.fit_transform(x_train)

    #creating a regression object
    polyreg_model = LinearRegression()

    #Now using 5-fold cross validation to obtain the best polynomial degree 
    r2_scores = cross_val_score(polyreg_model, x_train_poly, y_train, cv=5)
    
    #Get mean R-squared for a given polynomial degree 
    cv_scores[order] = np.mean(r2_scores)

#Selecting the best polynomial order 
best_order, best_score = None, None  
for order,score in cv_scores.items():
    if best_score is None or abs(best_score) < abs(score): 
        best_score = score 
        best_order = order 

#Reporting the best model with the most optimal polynomial 
print(f'The best model for the data has a polynomial degree of {best_order}, and R-squared score of: r2={round(best_score,3)}')
print('')



#Part Five: Hyperparameter Tuning 
#In this section I will perform hyperparameter tuning to control for the overfitting problem confronted and also check if in doing so the model's predictions
# can be improved even further. For hyperparameter tuning, I will be using Lasso regression regularization.

#Model Development: Polynomial Lasso Regression Model 

#Creating a pipeline to automate model development 
#Specifying the pipeline steps 
pipe_steps = [
                ('Polynomial', PolynomialFeatures()),      #performs a polynomial transform
                ('Model', Lasso())        #fits the data to lasso regression model    
                                    ]


#Creating the pipeline 
lasso_model = Pipeline(pipe_steps)


#Grid Search 
#Now performing grid search to obtain the best polynomial order and alpha value
#specifying the hyperparameters to test out (polynomial degrees & alpha values)
parameters = {'Polynomial__degree': [2,3,4],      #specifying the polynomials to test out
              'Model__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}     #specifying the alpha values to test out

#Creating a grid object and specifying the cross-validation characteristics
Grid = GridSearchCV(lasso_model, parameters, scoring='r2', cv=5) 

#Fitting the model with the training data for cross validation 
Grid.fit(x_train, y_train)


#Reporting the results of the cross validation (best polynomial order, alpha, and r2 score)
best_order = Grid.best_params_['Polynomial__degree']
best_alpha = Grid.best_params_['Model__alpha']
best_r2 = Grid.best_score_
print(f'The best model has a polynomial degree of {best_order}, alpha value of: alpha={best_alpha}, and r-squared score of r2={round(best_r2,3)}')
print('')


#Model Testing
#Now we can test the model one final time using the testing set 
#First, extracting the model with the best hyperparameters 
Lasso_Model = Grid.best_estimator_

#Calculating the R-squared score for the model using the testing set 
R2_test = Lasso_Model.score(x_test, y_test)
print(f'The r-squared score for the testing set is: r2={round(R2_test,3)}')
print('')


#Model Evaluation: Root Mean Squared Error 
#Generating price predictions using the testing set 
Y_pred_lasso = Lasso_Model.predict(x_test) 
#calculating root mean squared error for the testing set 
MSE = mean_squared_error(y_test, Y_pred_lasso)
RMSE = np.sqrt(MSE)
#Report the resulting RMSE value 
print(f'The root mean squared error is: RMSE={round(RMSE,3)}')
print('')


#Model Evaluation: Distribution Plot 
#Setting the characteristics of the plots 
ax1 = sns.distplot(y_test, hist=False, label='Actual Values')
sns.distplot(Y_pred_lasso, ax=ax1, hist=False, label='Predicted Values')
#Adding a title and labeling the axes
plt.title('Actual vs. Predicted Values for Car Prices\n(Polynomial Lasso Regression Model')
plt.xlabel('Car Price (in USD)', fontsize=12)
plt.ylabel('Distribution density of price values', fontsize=12)
plt.legend(loc='best')
#Adjusting the x-axis to display the prices in a reader-friendly format
plt.gcf().axes[0].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))
plt.xticks(rotation=90)

#Displaying the distribution plot
plt.show()


#Model Comparison 
#Setting the characteristics of the plots 
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12,6))
#Visualizing model fitting for the training set 
ax1 = sns.distplot(y_test, hist=False, ax=axes[0], label='Actual Values')
sns.distplot(Y_pred, hist=False, ax=ax1, label='Predicted Values')
#Visualizing model fitting for testing set 
ax2 = sns.distplot(y_test, hist=False, ax=axes[1], label='Actual Values')
sns.distplot(Y_pred_lasso, hist=False, ax=ax2, label='Predicted Values')

#Adding titles and labeling the axes 
fig.suptitle('Multiple Regression vs. Polynomial Regression')
axes[0].set_title('Multiple regression model fitting on testing')
axes[0].set_xlabel('Car Price (in USD)')
axes[0].set_ylabel('Distribution density of price values')
axes[0].legend(loc='best')
axes[1].set_title('Polynomial regression model fitting on testing')
axes[1].set_xlabel('Car Price (in USD)')
axes[1].set_ylabel('Distribution density of price values')
axes[1].legend(loc='best')
#Adjusting the x-axis to display the prices in a reader-friendly format
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
plt.gcf().axes[0].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))
plt.gcf().axes[1].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))

#show plot
plt.show()



#Part Six: Model Prediction
#In this section, as mentioned, I will build the final model again with the whole dataset and use it to generate predictions. 

#Final Model Development
#First, preparing the data for training the final model 
#specifying predictor variables 
x_data = df[['engine-size', 'curb-weight', 'horsepower', 'width', 'length', 'wheel-base', 'bore', 'city-mpg', 'highway-mpg',
            'make', 'num-of-cylinders', 'drive-wheels', 'fuel-system', 'engine-type', 'body-style', 'engine-location', 'aspiration']] 

#specifying the target variable
y_data = df['price']

#extracting categorical and numerical variables and storing them in separate objects 
# for processing them later separately
numerical_vars = x_data.select_dtypes(exclude='object').columns.tolist()
categorical_vars = x_data.select_dtypes(include='object').columns.tolist()


#Now building the pipeline 
#Creating the first part of the pipeline for normalizing numerical variables 
pipeline_pt1 = Pipeline([('Scaler', MinMaxScaler())])

#Creating the second part of the pipeline for encoding categorical variables
pipeline_pt2 = Pipeline([('Encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])

#Combining both pipelines 
pipeline_pt3 = ColumnTransformer([
                    ('NumScaler', pipeline_pt1, numerical_vars),
                    ('CatEncoder', pipeline_pt2, categorical_vars) 
                                                                   ])

#Adding a polynomial transform function to the pipeline 
pipeline_prep = Pipeline([('ColoumnTransformer', pipeline_pt3), 
                        ('Polynomial', PolynomialFeatures(degree=2))])


#Building the final pipeline for developing the polynomial lasso regression model  
Model = Pipeline([('Preprocessing', pipeline_prep),
                ('Model', Lasso(alpha=10))])

#Training the model with the entire dataset 
Model.fit(x_data, y_data)


#Now the model is ready and can be deployed for predictive pricing...


#Generating price predictions from novel data
#In this part, I will define a custom function, MakePrediction(), which will take novel new data of different car and employs the model to return the most 
# suitable price predictions based on these characteristics.
#Defining the function 
def MakePrediction(model, X_vars): 
    """This function takes two inputs: 'model', which specifies the model to be used to generate the price predictions, 
    and 'X_vars', which specifies the car characteristics for each car to make the price prediction based on. It runs
    the prediction-making process and returns a table with the predicted prices for each house."""
    
    Y_pred = model.predict(X_vars)
    Y_pred_df = pd.Series(Y_pred, name='Predicted Prices').to_frame().apply(lambda series: series.apply(lambda price: '${:,.2f}'.format(price)))
    return Y_pred_df 


#For a quick test of the function, I will extract a random sample of 10 data points from the original dataset and pass them to the function, along with the 
# final model developed above. The function should return 10 price predictions as best suited to these 10 data points.

#Extracting a random sample from the dataset and assigning it to 'X_new'
X_new = x_data.sample(10)

#Previewing the sample
print(X_new)
print('')


#Now passing the data to the MakePrediction() function to get price predictions
print(MakePrediction(Model, X_new))
print('')


#Showing the car characteristics and the corresponding predicted prices together
#Reindex and add predicted prices to dataframe 
sample_and_prediction, sample_and_prediction['Predicted Prices'] = X_new.reset_index(drop=True), MakePrediction(Model, X_new)
print(sample_and_prediction)
print('')


#Generating price predictions from user input 
#Finally, for this last part I will define another custom function, MakePrediction_forUser(), 
# which takes user input with the all characteristics of the car they wish to predict the 
# price of and returns a price prediction that best suits these given characteristics.

#Defining the function 
def MakePrediction_forUser(model, x_data):
    """This function asks the user for 17 inputs for 17 different car attributes: 
        Car brand - engine size - horsepower - fuel system - city MPG - highway MP -, engine type - engine location - 
        number of cylinders - curb weight - length - width - body style - drive wheels type - wheelbase - bore size - 
        aspiration engine type.
        
        After taking user input, the function returns a price prediction as best suited for these characteristics."""

    #create empty dictionary for input values 
    X_vars = dict()

    #Take user input for car characteristics 
    while True:
        make = input('Enter car brand: ')
        if make not in x_data['make'].unique().tolist(): 
            print('''Invalid car brand. This function only supports the list below:
                    [alfa-romero, audi, bmw, chevrolet, dodge, honda, isuzu, jaguar, mazda, mercedes-benz,
                     mercury, mitsubishi, nissan, peugot, plymouth, porsche, renault, saab, subaru, toyota,
                     volswagen, volvo].
                Please make sure to select a car brand featured on this list.\n''')
            continue 
        break 
    while True:
        try:
            engine_size = float(input('Enter engine size: '))
            break 
        except:
            print('Invalid input. Engine size must be a numerical value. Try again...\n')
            continue 
    while True:
        try:
            horsepower = float(input('Enter horsepower: '))
            break 
        except:
            print('Invalid input. Horsepower must be a numerical value. Try again...\n')
            continue 
    while True:
        fuel_system = input('Enter type of fuel system: ')
        if fuel_system not in x_data['fuel-system'].unique().tolist():
            print('''Invalid fuel system. This function only supports the list below:
                    [1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdim spfi].
                Please make sure to select a fuel system featured on this list.\n''')
            continue 
        break 
    while True: 
        try:            
            city_mpg = float(input('Enter city mpg: '))
            break
        except:
            print('Invalid input. City mpg must be a numerical value. Try again...\n')
            continue
    while True:
        try:
            highway_mpg = float(input('Enter highway mpg: '))
            break
        except:
            print('Invalid input. Highway mpg must be a numerical value. Try again...')
            continue
    while True: 
        engine_type = input('Enter engine type: ')
        if engine_type not in x_data['engine-type'].unique().tolist():
            print('''Invalid engine type. This function only supports the list below:
                    [l, dohc, ohc, ohcf, ohcv, rotor].
                Please make sure to select an engine type featured on this list.\n''')
            continue 
        break 
    while True:
        engine_location = input('Enter engine location: ')
        if engine_location not in x_data['engine-location'].unique().tolist():
            print("Invalid engine location. Engine location must be either 'front' or 'rear'. Please try again...\n")
            continue 
        break 
    while True: 
        num_of_cylinders = input('Enter number of cylinders (as written number): ')
        if num_of_cylinders not in x_data['num-of-cylinders'].unique().tolist():
            print('''Invalid number of cylinders. This function only supports the values below:
                    [two, three, four, five, six, eight, twelve].
                Please make sure to select a number featured on this list.\n''')
            continue 
        break 
    while True: 
        try:
            curb_weight = float(input('Enter car curb weight: '))
            break
        except:
            print('Invalid input. Curb weight must be a numerical value. Try again...\n')
            continue
    while True:
        try:
            length = float(input('Enter car length: '))
            break
        except:
            print('Invalid input. Car length must be a numerical value. Try again...\n')
            continue
    while True:
        try:
            width = float(input('Enter car width: '))
            break
        except:
            print('Invalid input. Car width must be a numerical value. Try again...\n')
            continue
    while True: 
        body_style = input('Enter body style: ')
        if body_style not in x_data['body-style'].unique().tolist():
            print('''Invalid car body style. This function only supports the values below:
                    [convertible, hardtop, hatchback, sedan, wagon].
                Please make sure to select a body style featured on this list.\n''')
            continue 
        break 
    while True:
        drive_wheels = input('Enter type of drive wheels: ')
        if drive_wheels not in x_data['drive-wheels'].unique().tolist():
            print('''Invalid drive wheels type. This function only supports the values below:
                    [4wd, fwd, rwd].
                Please make sure to select a drive wheel type featured on this list.\n''')
            continue 
        break 
    while True:
        try:
            wheel_base = float(input('Enter wheel base distance: '))
            break
        except:
            print('Invalid input. Wheel base distance must be a numerical value. Try again...\n')
            continue
    while True: 
        try:
            bore = float(input('Enter bore size: '))
            break 
        except:
            print('Invalid input. Bore size must be a numerical value. Try again...\n')
    while True:
        aspiration = input('Enter aspiration engine type: ')    
        if aspiration not in x_data['aspiration'].unique().tolist():
            print("Invalid aspiration engine type. This function only supports 'std' and 'turbo'. Please try again...")
            continue 
        break 

    print('\n\n')

    #Adding values to dictionary
    X_vars['engine-size'], X_vars['horsepower'], X_vars['city-mpg'],  X_vars['highway-mpg']  = [engine_size], [horsepower], [city_mpg], [highway_mpg] 
    X_vars['length'], X_vars['width'], X_vars['curb-weight'], X_vars['wheel-base'], X_vars['bore'] = [length], [width], [curb_weight], [wheel_base], [bore]
    X_vars['make'], X_vars['fuel-system'], X_vars['engine-type'], X_vars['engine-location'] = [make], [fuel_system], [engine_type], [engine_location]
    X_vars['num-of-cylinders'], X_vars['body-style'], X_vars['drive-wheels'], X_vars['aspiration'] = [num_of_cylinders], [body_style], [drive_wheels], [aspiration]
    
    #convert dictionary to dataframe 
    df_X_vars = pd.DataFrame(X_vars)

    #Generate and return price prediction
    Y_pred =  model.predict(df_X_vars)
    return 'For a car with the given characteristics, the predicted price is: ${:,.2f}.'.format(Y_pred[0])


#Now we can use the function to produce a prediction from user input 
print(MakePrediction_forUser(Model, x_data))

#END