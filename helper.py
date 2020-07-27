""" 
Created Date: July, 25 2020

This file contains all the functions which help in processing the luigi 
workflow. The file will be imported and reused as needed. Please visit function 
docstring for additional insights on inputs and outputs for each.

Please update the below list as and when adding new functions. 
The module consists of below funtions:
1. readData
2. cleanTweetData
3. calcCartesianCoord
4. buildTree
5. closestCity
6. createTrainData
7. trainModel
8. scoreData

"""
# Import core libraries
import pandas as pd
import numpy as np
import math
from scipy import spatial
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


def readData(file_path):
    """ 
    This function reads the file given its path. 

    Parameters:
    file_path: The file path for input file

    Returns:
    pandas.DataFrame: contents of file loaded onto DataFrame 
    """
    df = pd.read_csv(file_path, encoding = 'utf8')

    return df

def cleanTweetData(df):
    """ 
    This function cleans the tweet dataframe. It drops all the columns with 
    null and (0,0) coordinates. Also, we are spliting the tweet_coord variable
    into two variables tweet_lat, tweet_long respectively.

    Parameters:
    df: The tweet DataFrame

    Returns:
    pandas.DataFrame: cleaned tweet DataFrame 
    """
    df.dropna(subset = ["tweet_coord"], inplace = True) # drop rows with Null
    
    # Lets split the tweet_coord so that its easier to process
    df.tweet_coord = df.tweet_coord.str.strip('[]').str.split(', ')
    df[['tweet_lat', 'tweet_long']] = pd.DataFrame(df.tweet_coord.tolist(), \
                                                index= df.index)
    # Update dtype for lat and long to float
    df = df.astype({'tweet_lat': float, 'tweet_long': float})
    
    # Finally, update the df inplace to drop (0,0) lat/long 
    df.drop(df[(df['tweet_lat'] == 0) & (df['tweet_long'] == 0)].index, \
            inplace = True)

    return df

def calcCartesianCoord(latitude, longitude, elevation = 0):
    """ 
    This function calculates the cartesian coordinates for a given lat/long.

    Parameters:
    latitude: Input latitude
    longitude: Input longitude 
    elevation: default 0

    Returns:
    Tuple: Cartesian coordinates 
    """
    latitude = latitude * (math.pi/ 180)
    longitude = longitude * (math.pi/ 180)

    R = 6371 # Radius of Earth

    X = R * math.cos(latitude) * math.cos(longitude)
    Y = R * math.cos(latitude) * math.sin(longitude)
    Z = R * math.sin(latitude)

    return (X, Y, Z)

def buildTree(inp_data):
    """ 
    This function builds the KDTree which can be considered as a version of 
    Binary Search Tree. The tree itself is built only once and used for lookup
    for each lat/long.

    Parameters:
    Inp_data: The dataframe which contains (lat/long) coordinates which need to
                be represented on the tree.
    
    Returns:
    Tree: scipy.spatial.kdtree.KDTree 
    """

    places = []

    for index, row in inp_data.iterrows():
        coord = [row['latitude'], row['longitude']]
        cart_coord = calcCartesianCoord(*coord)
        places.append(cart_coord)

    tree = spatial.KDTree(places)

    return tree

def closestCity(lat, lon, tree, city_df):
    """ 
    This function calculates the closest lat/long pair by querying on the tree.
    Default distance metric used is euclidean ( denoted by p = 2)

    Parameters:
    lat: latitude of the input coordinate
    long: longitude of the input coordinate
    tree: scipy.spatial.kdtree.KDTree
    city_df: Cities dataframe used for look up based on index. 

    Returns:
    name: city name which is closest to the lat/lon pair. 
    """
    cart_coord = calcCartesianCoord(lat,lon)
    closest = tree.query([cart_coord], p = 2)
    index = closest[1][0]

    return city_df.name[index]

def createTrainData(tweet_df, cities_df):
    """ 
    This function creates the training data dataframe which will be input for 
    training the model. 

    Parameters:
    tweet_df: Input tweet dataframe
    cities_df: Input Cities dataframe

    Returns:
    train_df: Cleaned and encoded training data.
    encoder: 'OneHotEncoder' object which will be needed to encode real time 
            data. 
    """

    #Let us define the encoder object which will create x variables using 
    #all unique cities in cities.csv 
    
    encoder = OneHotEncoder(categories = [cities_df['name'].values], \
                            sparse = False, handle_unknown = 'ignore')
    #encoder.fit(tweet_df[['name']])

    train_x = encoder.fit_transform(tweet_df[['name']])
    col_names = encoder.get_feature_names([''])

    train_df =  pd.DataFrame(train_x, columns= col_names)
    train_df['label']= tweet_df['label'].values

    train_df['label']= LabelEncoder().fit_transform(train_df['label'])

    return train_df, encoder 

def trainModel(features_df):
    """ 
    This function creates a random forest classifier for the scenario based on 
    the input features_df. 

    Parameters:
    features_df: Cleaned training data with label column and one-hot encoded 
                city names.
    
    Returns:
    model: RandomForestClassifier object.
    """
    model = RandomForestClassifier(n_estimators = 10, \
                                        criterion = 'entropy', \
                                        random_state = 99)    

    x_train = features_df.drop(['label'],axis =1).values
    y_train = features_df['label'].values
    
    model.fit(x_train, y_train)

    return model

def scoreData(Inp_df, model, encoder):
    """ 
    This function scores the input dataframe with cities and outputs a dataframe
    with probabilities for each class in the label. 

    Parameters:
    Inp_df: Input Cities dataframe which consists of city names.
    encoder: encoder object which will be used to encode city names.
    model: model object which will be utilized for inferencing.

    Returns:
    score_df: A dataframe with probabilities for each of the cities.
    """
    test_x = encoder.transform(Inp_df[['name']])

    preds = model.predict_proba(test_x)

    #Create a dataframe to hold the predictions and add name column 
    score_df = pd.DataFrame(preds)
    score_df.insert(0, 'name', Inp_df['name'])

    # Rename the columns 
    score_df.rename(columns={0: 'Negative Probability', \
                             1: 'Neutral Probability', \
                             2: 'Positive Probability'}, \
                    inplace=True)
    
    score_df.sort_values(by = ['Positive Probability'], \
                         ascending = False, inplace = True)

    return score_df



