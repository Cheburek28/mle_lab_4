import pandas as pd
from pyspark.conf import SparkConf
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

from data_mart import DataMart

from logger import Logger

SHOW_LOG = True


class LogisticRegressionModel:
    def __init__(self):
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)

        self.datamart = DataMart()

        self.spark = SparkSession.builder \
            .master("local[4]") \
            .appName("NaiveBayesModel") \
            .getOrCreate()

    def train(self):
        df_train = self.datamart.get_data_test_classified_by_nb()

        sparkDFTrain = self.spark.createDataFrame(df_train)
        sparkDFTrain = sparkDFTrain.drop("label")
        sparkDFTrain = sparkDFTrain.withColumnRenamed("prediction", "label")
        self.log.info("Data loaded!")

        cols = sparkDFTrain.columns.copy()
        cols.remove("label")
        assemble = VectorAssembler(inputCols=cols, outputCol='features')
        sparkDFTrain = assemble.transform(sparkDFTrain)

        scale = StandardScaler(inputCol='features', outputCol='standardized', )
        data_scale = scale.fit(sparkDFTrain)
        sparkDFTrain = data_scale.transform(sparkDFTrain)

        lr = LogisticRegression()
        lr = lr.fit(sparkDFTrain)

        print(lr.getWeightCol())

    def save_model(self, df: pd.DataFrame):
        """Saves weights data"""
        pass


if __name__ == "__main__":
    conf = SparkConf() \
        .setAppName("MLE lab 3") \
        .setMaster("local")

    sc = SparkContext(conf=conf).getOrCreate()

    nbm = LogisticRegressionModel()
    nbm.train()

    sc.stop()
