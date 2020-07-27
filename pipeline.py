""" Loblaws Solutions Engineering take home problem """

#Import core libraries
import time
import os
import joblib

#Import third party libraries
import luigi

#Import user defined modules
import helper

# Global parameter which captures current time and uses it in placing output 
# files. 
task_run_time = time.strftime("%Y%m%d-%H%M%S")

class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid 
        geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')
    required_cols = ['airline_sentiment', 'tweet_coord', \
                         'tweet_lat', 'tweet_long']
    task_run_time = time.strftime("%Y%m%d-%H%M%S")

    def requires(self):
        return []
    
    def output(self):
        return luigi.LocalTarget("./output/{}/" \
                    .format(task_run_time) + self.output_file)

    def run(self):

        df = helper.readData(self.tweet_file)
        
        df = helper.cleanTweetData(df)

        #Check if path exists before attempting to write the data.
        # Create the parent directory if it does not exist.
        if not os.path.exists('./output/{}' \
                                .format(task_run_time)):
            os.makedirs('./output/{}'.format(task_run_time)) 

        df[self.required_cols].to_csv(self.output().path, index = False)



class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"

        encoder object is also saved as an output
    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')
    output_pkl = luigi.Parameter(default='encoder.pkl')
    

    def requires(self):
        return CleanDataTask(self.tweet_file)

    def output(self):
        return {'features': luigi.LocalTarget("./output/{}/" \
                    .format(task_run_time) + self.output_file), \
                'encoder':luigi.LocalTarget("./output/{}/" \
                    .format(task_run_time) + self.output_pkl)}

    def run(self):

        if not os.path.exists('./output/{}' \
                                .format(task_run_time)):
            os.makedirs('./output/{}'.format(task_run_time)) 

        tweet_df = helper.readData(self.input().path)
        
        cities_df = helper.readData(self.cities_file)
        
        # Drop any null values in the city names so that we wont run into 
        # errors in the later stages

        cities_df.dropna(subset = ['name'], inplace = True)

        # In order to find the closest city, we can build the KDTree 
        # by utilizing the scipy implementation. 
                
        tree = helper.buildTree(cities_df)

        # Update a new column in the Tweet_file df, since it holds the output
        # label 

        tweet_df['name'] = tweet_df.apply(lambda x: \
                                    helper.closestCity(x['tweet_lat'], \
                                                        x['tweet_long'], \
                                                        tree, cities_df), \
                                                            axis = 1)

        
        tweet_df = tweet_df[['airline_sentiment', 'name']]

        tweet_df.rename(columns={'airline_sentiment': 'label'}, inplace=True)

        feature_df, encoder = helper.createTrainData(tweet_df, cities_df)

        feature_df.to_csv(self.output()['features'].path, index = False)

        joblib.dump(encoder, self.output()['encoder'].path)
        

class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')

    def requires(self):
        return TrainingDataTask(self.tweet_file)

    def output(self):
        return luigi.LocalTarget("./output/{}/" \
                    .format(task_run_time) + self.output_file)

    def run(self):

        if not os.path.exists('./output/{}' \
                                .format(task_run_time)):
            os.makedirs('./output/{}'.format(task_run_time)) 

        print(self.input())
        
        features_df = helper.readData(self.input()['features'].path)
        
        model = helper.trainModel(features_df)

        joblib.dump(model, self.output().path)


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')
    cities_file = luigi.Parameter(default='cities.csv')

    def requires(self):
        return {'inp1': TrainModelTask(self.tweet_file),
                'inp2': TrainingDataTask(self.tweet_file)}

    def output(self):
        return luigi.LocalTarget("./output/{}/" \
                    .format(task_run_time) + self.output_file)

    def run(self):

        if not os.path.exists('./output/{}' \
                                .format(task_run_time)):
            os.makedirs('./output/{}'.format(task_run_time)) 

        encoder = joblib.load(self.input()['inp2']['encoder'].path)
        
        model = joblib.load(self.input()['inp1'].path)

        cities_df = helper.readData(self.cities_file)
        
        score_df = helper.scoreData(cities_df, model, encoder)

        score_df.to_csv(self.output().path, index = False)


if __name__ == "__main__":
    luigi.run()
