
'''
RMIT University Vietnam
Course: COSC2429 Introduction to Programming
Semester: 2024B
Assignment: 3
Group Number: 7
Group Members: 
    Vo Nguyen Bao Ngoc (s3975091)
    Phyu Phyu Shinn Thant (s4022136)
    #### ADD YOUR NAMES AND SID HERE ####

# Created date: 29/08/2024
# Last modified date: 29/08/2024
'''


''' PART 1/2 '''


# In[0]: Importing necessary packages
# Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from sklearn.model_selection import KFold   
from statistics import mean
import joblib 
# Import end



# In[1]: Get the data and load it
raw_data = pd.read_csv(r'datasets/realtor-data.csv')



# In[3]: Data Exploration
# Data Exploration
# 3.1 Quick view of the data
print('\n____________ Dataset info ____________')
print(raw_data.info())              
print('\n____________ Some first data examples ____________')
print(raw_data.head(3)) 
print('\n____________ Counts on a feature ____________')
print(raw_data['LEGAL DOCUMENTS'].value_counts()) 
print('\n____________ Statistics of numeric features ____________')
print(raw_data.describe())    
print('\n____________ Get specific rows and cols ____________')     
print(raw_data.iloc[[0,5,48], [2, 5]] ) # Refer using column ID
 
# 3.2 Scatter plot b/w 2 features
if 0:
    raw_data.plot(kind="scatter", y="PRICE IN MILLION VND", x="NUMBER OF BEDROOMS", alpha=0.2)
    #plt.axis([0, 5, 0, 10000])
    plt.savefig('figures/scatter_1_feat.png', format='png', dpi=300)
    plt.show()      
if 0:
    raw_data.plot(kind="scatter", y="PRICE IN MILLION VND", x="AREA IN M2", alpha=0.2)
    #plt.axis([0, 5, 0, 10000])
    #plt.savefig('figures/scatter_2_feat.png', format='png', dpi=300)
    plt.show()

# 3.3 Scatter plot b/w every pair of features
if 0:
    from pandas.plotting import scatter_matrix   
    features_to_plot = ["PRICE IN MILLION VND", "NUMBER OF BEDROOMS", "NUMBER OF TOILETS", "AREA IN M2"]
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.savefig('figures/scatter_mat_all_feat.png', format='png', dpi=300)
    plt.show()

# 3.4 Plot histogram of 1 feature
if 0:
    from pandas.plotting import scatter_matrix   
    features_to_plot = ["PRICE IN MILLION VND"]
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.show()

# 3.5 Plot histogram of numeric features
if 0:
    #raw_data.hist(bins=10, figsize=(10,5)) #bins: no. of intervals
    raw_data.hist(figsize=(10,5)) #bins: no. of intervals
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.tight_layout()
    plt.savefig('figures/hist_raw_data.png', format='png', dpi=300) # must save before show()
    plt.show()

# 3.6 Compute correlations b/w features
corr_matrix = raw_data.corr(numeric_only=True)
#print(corr_matrix) # print correlation matrix
print('\n',corr_matrix["PRICE IN MILLION VND"].sort_values(ascending=False)) # print correlation b/w a feature and other features

# 3.7 Try combining features
raw_data["AREA PER ROOM"] = raw_data["AREA IN M2"] / raw_data["NUMBER OF BEDROOMS"] 
raw_data["TOTAL NUMBER OF ROOMS"] = raw_data["NUMBER OF BEDROOMS"] + raw_data["NUMBER OF TOILETS"] 
corr_matrix = raw_data.corr(numeric_only=True)
print(corr_matrix["PRICE IN MILLION VND"].sort_values(ascending=False)) # print correlation b/w a feature and other features
raw_data.drop(columns = ["AREA PER ROOM", "TOTAL NUMBER OF ROOMS"], inplace=True) # remove experiment columns
# Data Exploration ends



# In[4]: Data Preprocessing 
# Data Preprocessing
# Remove an unused feature
raw_data.drop(columns = ["street"], inplace=True) 

# Removing outlier rows in the price feature to improve the RMSE scores
price_feature = ['price']
for col in price_feature:
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = raw_data[col].quantile(0.25)
    Q3 = raw_data[col].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove the outlier rows
    raw_data = raw_data[(raw_data[col] >= lower_bound) & (raw_data[col] <= upper_bound)]

raw_data.info()


#%% Split training and test datasets
# Different methods in case of the need for a stratified sample
method = 1
# Method 1: Randomly select 20% of data for test set.
if method == 1: 
    from sklearn.model_selection import train_test_split
    # Random state to get the same training set everytime
    train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=42)
print('\n____________ Split training and test set ____________')     
print(len(train_set), "training +", len(test_set), "test examples")
print(train_set.head(4))

#%% Label Separation
train_set_labels = train_set["price"].copy()
train_set = train_set.drop(columns = "price") 
test_set_labels = test_set["price"].copy()
test_set = test_set.drop(columns = "price") 



#%% Define pipelines for processing data. 
# Define ColumnSelector: a transformer for choosing columns
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values         

# Set values of the numeric and categorical pipelines
num_feat_names = ['bed', 'bath', 'acre_lot', 'house_size'] 
cat_feat_names = ['brokered_by', 'status', 'city', 'state', 'zip_code'] 

# Create a categorical pipeline
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    # Copy=false for imputing in place
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)),
    # Convert categorical data to one hot vector values
    ('cat_encoder', OneHotEncoder()) 
    ])    

# Pipeline testing
if False:
    trans_feat_values = cat_pipeline.fit_transform(train_set)

    # Check the encoded features
    if False: 
        # Display categories for each feature
        print(cat_pipeline.named_steps['cat_encoder'].categories_) 
        # Get feature names after encoding
        print(cat_pipeline.named_steps['cat_encoder'].get_feature_names_out(cat_feat_names)) 
        print("No. of one-hot columns: " + str(cat_pipeline.named_steps['cat_encoder'].get_feature_names_out(cat_feat_names).shape[0]))
        # Convert to dense array for viewing
        print(trans_feat_values.toarray())

# Pipeline testing ends

# Numerical Pipeline 
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    # Impute NaN values with median calculation
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)), 
    # Scale features to zero mean and unit variance
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) 
    ])  
  
# Combine features transformed by two above pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  

# Convert categorical columns to string and handle NaN values
for col in cat_feat_names:
    train_set[col] = train_set[col].astype(str).fillna("NO INFO")
# Process training data
processed_train_set_val = full_pipeline.fit_transform(train_set)


""" 
Output processed feature values for a better understanding + visualization
"""
# Output processed feature values
print('\n____________ Processed feature values ____________')
# Display first three process rows
print(processed_train_set_val[[0, 1, 2], :].toarray())
# Print the shape of the processed data
print(processed_train_set_val.shape)  

# One hot vectors count
cat_encoder = cat_pipeline.named_steps['cat_encoder']
num_onehot_count = sum(len(categories) for categories in cat_encoder.categories_)

print('We have %d numeric features + %d one-hot vectors for categorical features.' % (len(num_feat_names), num_onehot_count))
# Save the pipeline for future use
joblib.dump(full_pipeline, r'models/full_pipeline.pkl')


#%% Create a DataFrame for processed data (visualization)
""" Visualiation with DataFrame - to comment out"""
if False:  # Set to False to skip DataFrame creation
    onehot_cols = []
    # Collect one-hot encoded column names
    for val_list in full_pipeline.named_steps['cat_pipeline'].named_steps['cat_encoder'].categories_: 
        # Extend for readability
        onehot_cols.extend(val_list.tolist()) 

    # Create column headers
    columns_header = train_set.columns.tolist() + onehot_cols
    for name in cat_feat_names:
        columns_header.remove(name) 

    # Create DataFrame from processed values
    processed_train_set = pd.DataFrame(processed_train_set_val.toarray(), columns=columns_header)

    # Output processed DataFrame information
    print('\n____________ Processed DataFrame ____________')
    print(processed_train_set.info())
    # Display the first few rows
    print(processed_train_set.head())  

# Data Preprocessing ends


''' PART 2/2 '''

# In[5]: STEP 5. TRAIN AND EVALUATE MODELS 
#region
# 5.1 Try LinearRegression model
# 5.1.1 Training: learn a linear regression hypothesis using training data 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(processed_train_set_val, train_set_labels)
print('\n____________ LinearRegression ____________')
print('Learned parameters: ', model.coef_, model.intercept_)

# 5.1.2 Compute R2 score and root mean squared error
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse      
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
        
# 5.1.3 Predict labels for some training instances
print("\nInput data: \n", train_set.iloc[0:9])
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))

# 5.1.4 Store models to files, to compare latter
#from sklearn.externals import joblib 
import joblib # new lib
def store_model(model, model_name = ""):
    # NOTE: sklearn.joblib faster than pickle of Python
    # INFO: can store only ONE object in a file
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'models/' + model_name + '_model.pkl')
def load_model(model_name):
    # Load objects into memory
    #del model
    model = joblib.load('models/' + model_name + '_model.pkl')
    #print(model)
    return model
store_model(model)


#%% 5.2 Try DecisionTreeRegressor model
# Training
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error
print('\n____________ DecisionTreeRegressor ____________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)
# Predict labels for some training instances
#print("Input data: \n", train_set.iloc[0:9])
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.3 Try RandomForestRegressor model
# Training (NOTE: may take time if train_set is large)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 5) # n_estimators: no. of trees
model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error
print('\n____________ RandomForestRegressor ____________')
r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(model)      
# Predict labels for some training instances
#print("Input data: \n", train_set.iloc[0:9])
print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.4 Try polinomial regression model
# NOTE: polinomial regression can be treated as (multivariate) linear regression where high-degree features x1^2, x2^2, x1*x2... are seen as new features x3, x4, x5... 
# hence, to do polinomial regression, we add high-degree features to the data, then call linear regression
# 5.5.1 Training. NOTE: may take a while 
from sklearn.preprocessing import PolynomialFeatures
poly_feat_adder = PolynomialFeatures(degree = 2) # add high-degree features to the data
train_set_poly_added = poly_feat_adder.fit_transform(processed_train_set_val)
new_training = 10
if new_training:
    model = LinearRegression()
    model.fit(train_set_poly_added, train_set_labels)
    store_model(model, model_name = "PolinomialRegression")      
else:
    model = load_model("PolinomialRegression")
# 5.4.2 Compute R2 score and root mean squared error
print('\n____________ Polinomial regression ____________')
r2score, rmse = r2score_and_rmse(model, train_set_poly_added, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 5.4.3 Predict labels for some training instances
print("\nPredictions: ", model.predict(train_set_poly_added[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))


#%% 5.5 Evaluate with K-fold cross validation 
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
#from sklearn.model_selection import cross_val_predict

#cv1 = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42); 
#cv2 = StratifiedKFold(n_splits=10, random_state=42); 
#cv3 = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42); 
print('\n____________ K-fold cross validation ____________')

run_new_evaluation = 0
if run_new_evaluation:
    from sklearn.model_selection import KFold, StratifiedKFold
    # NOTE: 
    #   + If data labels are float, cross_val_score use KFold() to split cv data.
    #   + KFold randomly splits data, hence does NOT ensure data splits are the same (only StratifiedKFold may ensure that)
    cv = KFold(n_splits=5,shuffle=True,random_state=37) # cv data generator: just a try to persist data splits (hopefully)

    # Evaluate LinearRegression
    model_name = "LinearRegression" 
    model = LinearRegression()             
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("LinearRegression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    # Evaluate DecisionTreeRegressor
    model_name = "DecisionTreeRegressor" 
    model = DecisionTreeRegressor()
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    # Evaluate RandomForestRegressor
    model_name = "RandomForestRegressor" 
    model = RandomForestRegressor(n_estimators = 5)
    nmse_scores = cross_val_score(model, processed_train_set_val, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    # Evaluate Polinomial regression
    model_name = "PolinomialRegression" 
    model = LinearRegression()
    nmse_scores = cross_val_score(model, train_set_poly_added, train_set_labels, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-nmse_scores)
    joblib.dump(rmse_scores,'saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')
else:
    # Load rmse
    model_name = "LinearRegression" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("\nLinearRegression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "DecisionTreeRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("DecisionTreeRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "RandomForestRegressor" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

    model_name = "PolinomialRegression" 
    rmse_scores = joblib.load('saved_objects/' + model_name + '_rmse.pkl')
    print("Polinomial regression rmse: ", rmse_scores.round(decimals=1))
    print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')
#endregion



# In[6]: STEP 6. FINE-TUNE MODELS 
# NOTE: this takes TIME
#region
# IMPORTANT NOTE: since KFold split data randomly, the cv data in cross_val_score() above are DIFFERENT from SearchCV below.
#      => Should only compare resutls b/w SearchSV runs (NOT with cross_val_score()). 
# INFO: find best hyperparams (param set before learning, e.g., degree of polynomial in poly reg, no. of trees in rand forest, no. of layers in neural net)
# Here we fine-tune RandomForestRegressor and PolinomialRegression
print('\n____________ Fine-tune models ____________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    #print('Best estimator: ', grid_search.best_estimator_) # NOTE: require refit=True in  SearchCV
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params) 

method = 1
# 6.1 Method 1: Grid search (try all combinations of hyperparams in param_grid)
if method == 1:
    from sklearn.model_selection import GridSearchCV
    cv = KFold(n_splits=5,shuffle=True,random_state=37) # cv data generator
        
    run_new_search = 0      
    if run_new_search:        
        # 6.1.1 Fine-tune RandomForestRegressor
        model = RandomForestRegressor()
        param_grid = [
            # try 12 (3x4) combinations of hyperparameters (bootstrap=True: drawing samples with replacement)
            {'bootstrap': [True], 'n_estimators': [3, 15, 30], 'max_features': [2, 12, 20, 39]},
            # then try 12 (4x3) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [3, 5, 10, 20], 'max_features': [2, 6, 10]} ]
            # Train cross 5 folds, hence a total of (12+12)*5=120 rounds of training 
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True, 
        refit=True) # refit=True: after finding best hyperparam, it fit() the model with whole data (hope to get better result)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,'saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")      

        # 6.1.2 Fine-tune Polinomial regression  
        model = Pipeline([ ('poly_feat_adder', PolynomialFeatures()), # add high-degree features
                        ('lin_reg', LinearRegression()) ]) 
        param_grid = [
            # try 3 values of degree
            {'poly_feat_adder__degree': [1, 2, 3]} ] # access param of a transformer: <transformer>__<parameter> https://scikit-learn.org/stable/modules/compose.html
            # Train across 5 folds, hence a total of 3*5=15 rounds of training 
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', return_train_score=True)
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,'saved_objects/PolinomialRegression_gridsearch.pkl') 
        print_search_result(grid_search, model_name = "PolinomialRegression") 
    else:
        # Load grid_search
        grid_search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")         
        grid_search = joblib.load('saved_objects/PolinomialRegression_gridsearch.pkl')
        print_search_result(grid_search, model_name = "PolinomialRegression") 

# 6.2 Method 2: [EXERCISE] Random search n_iter times 
elif method == 2:
    from sklearn.model_selection import RandomizedSearchCV
    # ADD YOUR CODE HERE
    
#endregion



# In[7]: STEP 7. ANALYZE AND TEST YOUR SOLUTION
# NOTE: solution is the best model from the previous steps. 
#region
# 7.1 Pick the best model - the SOLUTION
# Pick Random forest
search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
best_model = search.best_estimator_
# Pick Linear regression
#best_model = joblib.load('saved_objects/LinearRegression_model.pkl')

print('\n____________ ANALYZE AND TEST YOUR SOLUTION ____________')
print('SOLUTION: ' , best_model)
store_model(best_model, model_name="SOLUTION")   

# 7.2 Analyse the SOLUTION to get more insights about the data
# NOTE: ONLY for random forest
if type(best_model).__name__ == "RandomForestRegressor":
    # Print features and importance score  (ONLY on rand forest)
    feature_importances = best_model.feature_importances_
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    feature_names = train_set.columns.tolist() + ["TOTAL NUMBER OF BEDROOMS"] + onehot_cols
    for name in cat_feat_names:
        feature_names.remove(name)
    print('\nFeatures and importance score: ')
    print(*sorted(zip( feature_names, feature_importances.round(decimals=4)), key = lambda row: row[1], reverse=True),sep='\n')

#%% 7.3 Run on test data
full_pipeline = joblib.load(r'models/full_pipeline.pkl')
processed_test_set = full_pipeline.transform(test_set)  
# 7.3.1 Compute R2 score and root mean squared error
r2score, rmse = r2score_and_rmse(best_model, processed_test_set, test_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 7.3.2 Predict labels for some test instances
print("\nTest data: \n", test_set.iloc[0:9])
print("\nPredictions: ", best_model.predict(processed_test_set[0:9]).round(decimals=1))
print("Labels:      ", list(test_set_labels[0:9]),'\n')

print('''       
Reasons for low performance (coined from data):
    - Data inaccuracy: Some samples might not be apartments, or there might be errors in Area, Number of bedrooms, etc.
    - Label inconsistencies: Housing prices can vary depending on the real estate agent and the time of the listing (multiple prices for the same property).
    - Noisy data: Outliers like properties with extremely large square footage can distort the results.
    - Insufficient data: Consider reducing the number of features (e.g., orientation, district) or increasing the sample size.
      
Suggested improvements:
    - Remove outliers: Eliminate samples with unusually high or low prices.
    - Dimensionality reduction: such as removing the "ORIENTATION" column, since this feature might not be contributing significantly to the model's accuracy.
    - Add more samples: Increase the sample size to improve the model's generalization ability.
''')
#endregion



# In[8]: STEP 8. LAUNCH, MONITOR, AND MAINTAIN YOUR SYSTEM
# Go to slide: see notes

done = 1