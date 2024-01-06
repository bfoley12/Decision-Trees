import pandas as pd
import numpy as np

# Loading Data
def load_from_csv(option):
    """ Loads in the data from the UCI Machine Learning Repository

    This is extenisble by adding options and their relative (or absolute) paths to the 'paths' variable. The numbers coincide with the numbers from Assignment 1.
    Option 6 was a special case of read in, as it was a .csv with headers as opposed to a .data file lacking headers.
    I also added headers and modified the data as noted in Assignment 1. I left imputation and conversion to outside calls.

    Args:
        option (int): An option of the dataset to load (1 indexed, as Assignment 1 had it)
    
    Returns:
        data (dataframe): a Pandas DataFrame representing the read in and modified data
        0 : returend when an improper option is given
    """
    
    paths = ["C:/Users/brend/Documents/Classes/Machine Learning//DataSets//Breast Cancer//breast-cancer-wisconsin.data", "C:/Users/brend/Documents/Classes/Machine Learning/DataSets/Cars/car.data", 
             "C:/Users/brend/Documents/Classes/Machine Learning/DataSets/Congress/house-votes-84.data", "C:/Users/brend/Documents/Classes/Machine Learning/DataSets/Abalone/abalone.data", 
        "C:/Users/brend/Documents/Classes/Machine Learning/DataSets/Machine/machine.data", "C:/Users/brend/Documents/Classes/Machine Learning/DataSets/Forest Fires/forestfires.csv"]
    if option == 6:
        data = pd.read_csv(paths[option - 1])
    else:
        data = pd.read_csv(paths[option - 1], header = None)
    match option:
        case 1:
            data.columns = ["id", "clump_thickness", "size_uniformity", "shape_uniformity", "marginal_adhesion", "epithelial_size", "bare_nuclei", "bland_chromatin", 
                "normal_nucleoli", "mitoses", "class"]
            data = impute(data, range(11))
            for col in data.columns:
                data[col] = data[col].astype('category')
        case 2:
            data.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
            data = ordinal_conversion(data, range(7))
            for col in data.columns:
                data[col] = data[col].astype('category')
        case 3:
            data.columns = ["class", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid", 
                "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback",
                "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"]
            data = pd.concat([data, nominal_conversion(data, range(1, 17))], axis = 1)
            data.drop(data.columns[range(1, 17)], inplace = True, axis = 1)
            data = ordinal_conversion(data, [0])
            for col in data.columns:
                data[col] = data[col].astype('category')
        case 4:
            data.columns = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "class"]
            data = ordinal_conversion(data, [0])
            data = impute(data, range(1, 9))
            data["sex"] = data["sex"].astype('category')
        case 5:
            data.columns = ["name", "model", "myct", "mmin", "mmax", "cach", "chmin", "chmax", "class", "erp"]
            # data = pd.concat([data, nominal_conversion(data, [0, 1])], axis = 1)
            data.drop(["name", "model"], axis = 1, inplace = True)
            data = impute(data, range(8))
        case 6:
            # Log transformation of area to correct for 0.0 skew, as suggested by authors
            # Adjusted by 0.01 to avoid log(0)
            data["area"] = np.log(data["area"] + 0.01)
            data = ordinal_conversion(data, [2,3])
            data = impute(data, range(5, 13))
            data.rename(columns = {"area": "class"}, inplace = True)
            for col in data.columns[0:4]:
                data[col] = data[col].astype('category')
        case _:
            return 0
    return data

def impute(df, indicator):
    """Imputes data on specified columns of a dataframe
    
    Given a dataframe and the speicifed columns, computes column-wise means and imputes missing values
    
    Args:
        df (dataframe): the dataframe with missing values to compute
        indicator (int list): list of column indices to impute on
    
    Returns:
        df (dataframe): the dataframe with completed mean imputation 
    """

    temp_df = df.iloc[:, indicator]
    temp_df = temp_df.replace("?", np.NaN)
    temp_df = temp_df.apply(lambda x: pd.to_numeric(x,errors='ignore'))
    df.iloc[:, indicator] = temp_df.fillna(temp_df.mean(numeric_only = True))
    return df

def ordinal_conversion(df, indicator):
    """Converts ordinal data into integers
    
    Given a dataframe and the specified columns, converts ordinal data into integers
    
    Args:
        df (dataframe): the dataframe with at least one ordinal column
        indicator (int list): list of ordinal column indices
    
    Returns:
        df (dataframe): the dataframe with integer-converted ordinal columns   
    """

    df.iloc[:, indicator] = df.iloc[:, indicator].rank(method = "dense").astype(int)
    return df

# Need to test on something better than option 4
def nominal_conversion(df, indicator):
    """Converts nominal data into integers
    
    Given a dataframe and the specified columns, converts nominal data into integers
    
    Args:
        df (dataframe): the dataframe with at least one nominal column
        indicator (int list): list of nominal column indices
    
    Returns:
        df (dataframe): the dataframe with integer-converted nominal columns    
    """

    return pd.get_dummies(df.iloc[:, indicator], prefix = list(df.columns[indicator]))

# Note: "equal" frequency may be off due to repeating values in measurement. Introducing noise will return a more equal bin distribution
def discretize(df, indicator, num_bins = 10, width_freq = 1):
    """Discretizes dataframe columns based on a specified number of bins to equal width or frequency
    
    Args:
        df (dataframe): the dataframe to perform discretization on
        indicator (int list): list of column indices to discretize
        num_bins (int): number of bins to pool values into
        width_freq (bool): true - equal width discretization; false - equal frequency discretization
        
    Returns:
        df (dataframe): the dataframe with discretized columns
    """
    labels = range(num_bins)
    if width_freq:
        for i in range(len(indicator)):
            df.iloc[:, indicator[i]] = pd.cut(df.iloc[:, indicator[i]], bins = num_bins, labels = labels).astype(int)
    else:
        for i in range(len(indicator)):
            df.iloc[:, indicator[i]] = pd.qcut(df.iloc[:, indicator[i]], q = num_bins, labels = labels).astype(int)
    return df

def standardize(training, test):
    """Standardizes the test and training set
    
    Computes the Z-score of every column based on the training set and standardizes the training and test sets
    
    Args:
        training (dataframe): the training set to standardize
        test (dataframe): the test set to standardize
        
    Returns:
        tuple (dataframe, dataframe): a tuple of the training and test sets post-standardization
    """

    m = training.mean()
    s = training.std()
    return ((training - m) / s, (test - m) / s)

def cross_validation(df, k, classification, key_column = "", k_by_two = 1):
    """Performs Cross-Validation on a dataset
    
    Args:
        df (dataframe): the dataframe to generate training, test, and validation sets from
        key_column (string): names of class column to stratify classification data with
        
    Returns:
        tuple (dataframe, dataframe, dataframe): a tuple of (training, test, validation) datasets
    """
    res = []
    if k_by_two:
        for i in range(int(k/2)):
            if key_column:
                training = df.groupby(key_column, group_keys = False).apply(lambda g: g.sample(frac=0.8))
                validation = df.loc[~df.index.isin(training.index)]
                test = training.groupby(key_column, group_keys = False).apply(lambda g: g.sample(frac = .5))
                training = training.loc[~training.index.isin(test.index)]
            else:
                training = df.sample(frac = 0.8)
                validation = df.loc[~df.index.isin(training.index)]
                test = training.sample(frac = .5)
                training = training.loc[~training.index.isin(test.index)]
            training, test = standardize(training, test)
            res.append(list(null_model(training, test, classification, key_column).values())[0])
        tune = max(res)
        res = []
        for i in range(int(k/2)):
            if key_column:
                training = df.groupby(key_column, group_keys = False).apply(lambda g: g.sample(frac=0.8))
                validation = df.loc[~df.index.isin(training.index)]
                test = training.groupby(key_column, group_keys = False).apply(lambda g: g.sample(frac = .5))
                training = training.loc[~training.index.isin(test.index)]
            else:
                training = df.sample(frac = 0.8)
                validation = df.loc[~df.index.isin(training.index)]
                test = training.sample(frac = .5)
                training = training.loc[~training.index.isin(test.index)]
            training, test = standardize(training, test)
            res.append(list(null_model(training, test, classification, key_column).values())[0])
            res.append(list(null_model(test, training, classification, key_column).values())[0])
    return sum(res)/len(res)

def evaluate(ground_truth, prediction, classification):
    """Generates statistics on prediction results
    
    Based on a given ground truth, assesses quality of prediction.
    Currently only have accuracy and MSE for classification and regression, respectively, but is easily extensible to other scores
    
    Args:
        ground_truth (list): list of the correct classifications to compare against
        predicition (list): list of predictions made by model to compare against the ground truth
        classification (boolean): whether the task is classification (true) or regression (false)
        
    Returns:
        dictionary: dictionary of various assessments of prediction quality, depending on the task (classification or regression)'
    """

    ground_truth = pd.Series(ground_truth)
    ground_truth.reset_index(inplace = True, drop = True)
    prediction = pd.Series(prediction)
    if classification:
        df = {"Accuracy": (ground_truth == prediction).sum()/len(ground_truth)}
    else:
        df = {"MSE": ((ground_truth - prediction)**2).mean()}
    return df

def null_model(training, test, classification, key_column):
    """Naive algorithm for prediction
    
    Looks at the majority class label (for classification) or the average of scores (for regression) in training set and checks
    validity against the test set
    
    Args:
        training (dataframe): training data
        test (dataframe): test data
        classification (boolean): true - classification task; false - regression task
        key_column (string): name of column to use for prediction
        
    Returns:
        dictionary: a dictionary of statistics for fit of the learned rules on the test data, depending on task (classification or regression)
    """

    if classification:
        pred = [training[key_column].mode()[0]] * len(test)
    else:
        pred = [training[key_column].mean()] * len(test)
    return evaluate(test[key_column], pred, classification)
