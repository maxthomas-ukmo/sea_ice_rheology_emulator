import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pickle
from sklearn.preprocessing import StandardScaler
from make_feature_label_pairs import make_feature_label_pairs
import random 
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import pandas as pd
import argparse as ap
import warnings
from sklearn.exceptions import DataConversionWarning
import pprint

# Define functions

# Parse command line arguments
def parse_args(dummy_args=False):

    if dummy_args:
        args = {'model_type': 'SGDRegressor',
            'test_fraction': 0.1,
            'suite': 'u-cn464',
            'version': 'raw_v0',
            'features': ['siconc', 'sithic', 'utau_ai', 'utau_oi', 'vtau_ai', 'vtau_oi'],#,'sishea', 'sistre'],
            'labels': ['sig1_pnorm'],
            'flatten': True,
            'StandardScalar': True
               }
        return args
    else:
        parser = ap.ArgumentParser(description='Train a regression model')
        parser.add_argument('--model_type', type=str, default='SGDRegressor', help='Type of model to use')
        parser.add_argument('--test_fraction', type=float, default=0.1, help='Fraction of data to use for testing')
        parser.add_argument('--suite', type=str, default='u-cn464', help='Name of suite to use')
        parser.add_argument('--version', type=str, default='raw_v0', help='Version of suite to use')
        parser.add_argument('--features', type=str, default='siconc,sithic,utau_ai,utau_oi,vtau_ai,vtau_oi,sishea,sistre,sig1_pnorm,sig2_pnorm,sivelv,sivelu', help='List of features to use')
        parser.add_argument('--labels', type=str, default='sig1_pnorm', help='List of labels to use')
        parser.add_argument('--flatten', type=bool, default=True, help='Flatten data')
        parser.add_argument('--StandardScalar', type=bool, default=True, help='Standardise data')
        parser.add_argument('--tune', type=bool, default=True, help='Tune hyperparameters')
        parser.add_argument('--data_points', type=int, default=-1, help='Data points to feed model (less for quick testing)')
        parser.add_argument('--random_seed', type=int, default=1, help='Random seed')
        parser.add_argument('--validation_fraction', type=float, default=0.2, help='Fraction of train data to validate on')
        parser.add_argument('--testing_pair_choice', type=int, default=0, help='Choice of pair to use for quick code development')

        args = parser.parse_args()
        args.features = args.features.split(',')     
        args.labels = args.labels.split(',')         
                                                                                                                                                                                                                                                     
        return vars(args)

# Load data function
def load_data(args):
    suite = args['suite']
    version = args['version']
    filename = '../data/' + suite + '/raw/' + suite + '_' + version + '.nc'
    pairs = make_feature_label_pairs(filename, args['features'], args['labels'], flatten=args['flatten'])
    return pairs

# Make test and train sets
def make_test_train(pairs, args):
    # convert pairs to pandas dataframe
    pairs_df = []
    for pair in pairs:
        pairs_df.append( 
                         (pd.DataFrame(pair[0].to_array().values.T), 
                          pd.DataFrame(pair[1].to_array().values.reshape(-1, ))) 
                        )

    # randomly select 10% of the pairs as a test holdout
    n = len(pairs_df)
    n_test = int(args['test_fraction']*n)
    test_indices = random.sample(range(n), n_test)
    train_indices = [i for i in range(n) if i not in test_indices]

    train_pairs = [pairs_df[i] for i in train_indices]
    test_pairs = [pairs_df[i] for i in test_indices]

    train_X = pd.concat([pair[0] for pair in train_pairs], axis=0)
    train_y = pd.concat([pair[1] for pair in train_pairs], axis=0)
    test_X = pd.concat([pair[0] for pair in test_pairs], axis=0)
    test_y = pd.concat([pair[1] for pair in test_pairs], axis=0)
    
    print('Train shape initial: ', train_X.shape, train_y.shape)
    print('Test shape initial: ', test_X.shape, test_y.shape)

    return train_X, train_y, test_X, test_y

def reduce_data_points(train_X, train_y, n_data_points):
    if n_data_points > 0:
        indicies = random.sample(range(len(train_X)), n_data_points)
        train_X = train_X.iloc[indicies]
        train_y = train_y.iloc[indicies]
    return train_X, train_y

def make_validation(train_X, train_y, validation_fraction):
    n = len(train_X)
    n_val = int(validation_fraction*n)
    val_indices = random.sample(range(n), n_val)
    train_indices = [i for i in range(n) if i not in val_indices]

    val_X = train_X.iloc[val_indices]
    val_y = train_y.iloc[val_indices]
    train_X = train_X.iloc[train_indices]
    train_y = train_y.iloc[train_indices]

    print('Train shape reduced: ', train_X.shape, train_y.shape)
    print('Validation shape: ', val_X.shape, val_y.shape)

    return train_X, train_y, val_X, val_y

# Make parameter grid
def make_param_grid(args):
    if args['model_type'].lower() == 'sgdregressor':
        param_grid = {
                    'sgdregressor__alpha': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                    'sgdregressor__loss': ['squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'],
                    'sgdregressor__penalty': ['elasticnet'],
                    'sgdregressor__l1_ratio': [0, 0.05, 0.15, 0.2, 0.4, 0.6, 0.8, 1]
                    }
    elif args['model_type'].lower() == 'decisiontreeregressor':
        param_grid = {'decisiontreeregressor__max_depth': [None, 10, 20, 30, 40, 50, 100],
                'decisiontreeregressor__min_samples_split': [2, 5, 10, 20],
                'decisiontreeregressor__min_samples_leaf': [1, 2, 4, 8],
                'decisiontreeregressor__min_impurity_decrease': [0.0, 0.25, 0.5],
                'decisiontreeregressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']}
    elif args['model_type'].lower() == 'randomforestregressor':
        param_grid = {'randomforestregressor__n_estimators': [100, 200, 300, 400, 500],
                    'randomforestregressor__max_depth': [10, 20, 30, 40, 50],
                    'randomforestregressor__min_samples_split': [2, 5, 10],
                    'randomforestregressor__min_samples_leaf': [1, 2, 4],
                    'randomforestregressor__max_features': ['auto', 'sqrt', 'log2']
                    }
        # param_grid = {'randomforestregressor__n_estimators': [10, 50, 100],
        #         'randomforestregressor__max_depth': [None, 2, 4, 6, 8, 10],
        #         'randomforestregressor__min_samples_split': [2, 6, 10],
        #         'randomforestregressor__min_samples_leaf': [1, 6, 9],
        #         'randomforestregressor__min_impurity_decrease': [0.0, 0.25, 0.5]}
    elif args['model_type'].lower == 'gradientboostingregressor':
        param_grid = {'gradientboostingregressor__n_estimators': [100, 250, 500, 750, 1000],
                'gradientboostingregressor__max_depth': [3, 5, 7, 9],
                'gradientboostingregressor__min_samples_split': [3, 6, 9, 12],
                'gradientboostingregressor__min_samples_leaf': [1, 4, 8, 12],
                'gradientboostingregressor__min_impurity_decrease': [0.0, 0.1, 0.5]}
    elif args['model_type'].lower() == 'xgbrregressor':
        param_grid = {'xgbrregressor__n_estimators': [10, 100, 350, 500, 1000],
                'xgbrregressor__max_depth': [2, 6, 10],
                'xgbrregressor__min_samples_split': [2, 6, 10],
                'xgbrregressor__min_samples_leaf': [1, 6, 9],
                'xgbrregressor__min_impurity_decrease': [0.0, 0.25, 0.5]}
    elif args['model_type'].lower() == 'gradientboostingregressor':
        param_grid = {'gradientboostingregressor__n_estimators': [10, 50, 100],
                'gradientboostingregressor__max_depth': [2, 3, 6, 10],
                'gradientboostingregressor__min_samples_split': [2, 6, 10],
                'gradientboostingregressor__min_samples_leaf': [1, 6, 9],
                'gradientboostingregressor__min_impurity_decrease': [0.0, 0.25, 0.5]}
    elif args['model_type'].lower() == 'svr':
        param_grid = {'svr__C': np.logspace(-4, 2, 10),
                'svr__gamma': np.logspace(-4, 2, 10),
                'svr__kernel': ['sigmoid', 'rbf']}
    elif args['model_type'].lower() == 'kernelridge':
        param_grid = {'kernelridge__alpha': np.logspace(-4, 2, 10),
                'kernelridge__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'kernelridge__gamma': np.logspace(-4, 2, 10)}
    elif args['model_type'].lower() == 'bayesianridge':
        param_grid = {'bayesianridge__alpha_1': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'bayesianridge__alpha_2': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'bayesianridge__lambda_1': [1e-05, 0.0001, 0.001, 0.01, 0.1],
                'bayesianridge__lambda_2': [1e-05, 0.0001, 0.001, 0.01, 0.1]}
    elif args['model_type'].lower() == 'lgbmregressor':
        param_grid = {'lgbmregressor__n_estimators': [10, 100, 350, 500, 1000],
                'lgbmregressor__max_depth': [2, 4, 6, 8, 10],
                'lgbmregressor__min_child_samples': [2, 4, 6, 8, 10],
                'lgbmregressor__min_child_weight': [1e-05, 0.0001, 0.01, 0.1],
                'lgbmregressor__subsample': [0.1, 0.5, 1],
                'lgbmregressor__colsample_bytree': [0.1, 0.5, 1]}
    elif args['model_type'].lower() == 'catboostregressor':
        param_grid = {'catboostregressor__n_estimators': [100, 300, 500],
                'catboostregressor__max_depth': [2, 6, 10],
                'catboostregressor__learning_rate': [0.1, 0.5, 1],
                'catboostregressor__l2_leaf_reg': [1e-05, 0.0001, 0.01, 0.1]}
    elif args['model_type'].lower() == 'logisticregression':
        param_grid = {'logisticregression__penalty' : ['elastic_net','none'],
                'logisticregression__C' : np.logspace(-4, 2, 10),
                'logisticregression__l1_ratio' : [0, 0.2, 0.4, 0.6, 0.8, 1]}
    return param_grid

# Load model class
def load_model(args):
    if args['model_type'].lower() == 'sgdregressor':
        from sklearn.linear_model import SGDRegressor
        model = SGDRegressor()
    elif args['model_type'].lower() == 'linearregressor':
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    elif args['model_type'].lower() == 'decisiontreeregressor':
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor()
    elif args['model_type'].lower() == 'randomforestregressor':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor()
    elif args['model_type'].lower() == 'gradientboostingregressor':
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor()
    elif args['model_type'].lower() == 'svr':
        from sklearn.svm import SVR
        return SVR()
    elif args['model_type'].lower() == 'kernelridge':
        from sklearn.kernel_ridge import KernelRidge
        return KernelRidge()
    elif args['model_type'].lower() == 'bayesianridge':
        from sklearn.linear_model import BayesianRidge
        return BayesianRidge()
    elif args['model_type'].lower() == 'logisticregression':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression()
    elif args['model_type'].lower() == 'decisiontreeclassifier':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier()
    elif args['model_type'].lower() == 'randomforestclassifier':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()
    elif args['model_type'].lower() == 'gradientboostingclassifier':
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier()
    else:
        raise ValueError(args['model_type'] + ' is not a valid model name')
    return model

# Set up model
def define_model(model, args):
    if args['StandardScalar']:
        scaler = StandardScaler()
        pipe = make_pipeline(scaler, model)
    else:
        pipe = model
    return pipe

# # Train baseline model
# def train_baseline_model(pipe, train):
#     pipe.fit(train[0], train[1])
#     return pipe

# Tune hyperparameters
def tune_hyperparameters(pipe, param_grid, train_X, train_y):        
    search = GridSearchCV(pipe, param_grid, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=5, verbose=1)
    search.fit(train_X, train_y)
    return search

# Evaluate model
def evaluate_model(pipe, val_X, val_y, train_y=None):
    y_pred = pipe.predict(val_X)
    score = mean_squared_error(val_y, y_pred)

    fig = plt.figure()
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(val_y, y_pred, s=0.2, color='black', alpha=0.5)
    ax1.plot([-1,1],[-1,1], color='black')
    ax1.set_title('Predicted vs. True, MSE = ' + str(score))

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(val_y, bins=100, density=True, color='blue', alpha=0.5, label='Validation y')
    ax2.hist(y_pred, bins=100, density=True, color='red', alpha=0.5, label='Predicted y')
    if train_y is not None:
        ax2.hist(train_y, bins=100, density=True, color='green', alpha=0.5, label='Train y')
    ax2.legend()
    return y_pred, score, fig

def main(args):

    warnings.simplefilter("ignore", category=DataConversionWarning)

    # Load in raw data and make pairs of y(t+1) and x(t)
    pairs = load_data(args)

    # Split data into a training (for train and validation) and testing set (not used for now)
    all_train_X, all_train_y, test_X, test_y = make_test_train(pairs, args)

    # For quick testing, reduce the number of data points (set args['data_points'] to the number of data points you want, or -1 to keep everything)
    reduced_train_X, reduced_train_y = reduce_data_points(all_train_X, all_train_y, args['data_points'])

    # Split the training data into a training and validation set
    train_X, train_y, val_X, val_y = make_validation(reduced_train_X, reduced_train_y, args['validation_fraction'])
    
    # Make parameter grid for tuning hyperparameters
    param_grid = make_param_grid(args)

    # Load desired model(args['model_type']) 
    model = load_model(args) # TODO: pass model_type rather than args

    # Define model by building a pipe that includes (or doesn't) a StandardScalar() step
    baseline_pipe = define_model(model, args) # TODO: pass standard scalar bool rather than args

    # Train baseline model using preset params
    print('Starting baseline')
    baseline = baseline_pipe.fit(train_X, train_y)
    # save baseline model
    with open('baseline_model.pkl', 'wb') as f:
        pickle.dump(baseline, f)

    # Evaluate baseline model
    # y_pred_baseline, score_baseline, fig_baseline = evaluate_model(baseline, train_X, train_y, val_y)
    val_pred = baseline.predict(val_X)
    train_pred = baseline.predict(train_X)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(val_y, val_pred, s=0.2, color='black', alpha=0.5, label='Validation')
    #ax.scatter(train_y, train_pred, s=0.2, color='red', alpha=0.5, label='Train')
    ax.set_title('Predicted vs. True ' + args['model_type'] + ': ' + str(mean_squared_error(val_y, val_pred))) 
    ax.plot([-1,0.2],[-1,0.2], color='black')
    ax.legend()

    fig.savefig('modelling-baseline.png')

    # If tuning, tune hyperparameters and evaluate best pipe
    if args['tune']:
        print('Starting tuning')
        tuning_pipe = define_model(model, args)

        hp_search = tune_hyperparameters(tuning_pipe, param_grid, train_X, train_y)
        # y_pred_tuned, score_tuned, fig_tuned = evaluate_model(hp_search.best_estimator_, val_X, val_y, train_y)
            
        fig_tuned = plt.figure()
        ax = fig_tuned.add_subplot(111)
        ax.scatter(val_y, hp_search.best_estimator_.predict(val_X), s=0.2, color='black', alpha=0.5, label='Validation')
        #ax.scatter(train_y, hp_search.best_estimator_.predict(train_X), s=0.2, color='red', alpha=0.5, label='Train')
        ax.set_title('Predicted vs. True ' + args['model_type'] + ': ' + str(mean_squared_error(val_y, hp_search.best_estimator_.predict(val_X)))) 

        ax.plot([-1,0.2],[-1,0.2], color='black')
        
        fig_tuned.savefig('modelling-tuned.png')
        # save search
        with open('hp_search.pkl', 'wb') as f:
            pickle.dump(hp_search, f)

        print('')
        print('==================================')
        print('Finished modelling')
        pprint(args)
        print('Baseline parameters: ', baseline.get_params())
        print('Baseline model MSE: ', mean_squared_error(val_y, baseline.predict(val_X)))
        print('Best hyperparameters: ', hp_search.best_estimator_.get_params())
        print('Best model MSE: ', mean_squared_error(val_y, hp_search.best_estimator_.predict(val_X)))
        print('==================================')

    # # Print results
    # print('Baseline model MSE: ', score_baseline)
    # print('Baseline parameters: ', baseline.get_params())

    # if args['tune']:
    #     print('Tuned model MSE: ', score_tuned)
    #     print('Best parameters: ', hp_search.best_estimator_.get_params())

    

if __name__ == '__main__':

    args = parse_args()

    # Set the random seed for reproducibility
    random.seed(args['random_seed'])

    main(args)