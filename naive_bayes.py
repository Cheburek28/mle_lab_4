import pandas as pd
from pyspark.conf import SparkConf
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from data_mart import DataMart

from logger import Logger

SHOW_LOG = True


class NaiveBayesModel:
    def __init__(self):
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)

        self.datamart = DataMart()

        self.spark = SparkSession.builder \
            .master("local[4]") \
            .appName("NaiveBayesModel") \
            .getOrCreate()

    def train(self):
        df_train = self.datamart.get_train_data()
        df_test = self.datamart.get_test_data()

        sparkDFTrain = self.spark.createDataFrame(df_train)
        sparkDFTrain = sparkDFTrain.withColumnRenamed("prediction", "label")
        sparkDFTest = self.spark.createDataFrame(df_test)
        sparkDFTest = sparkDFTest.withColumnRenamed("prediction", "label")
        self.log.info("Data loaded!")

        cols = sparkDFTrain.columns.copy()
        cols.remove("label")
        assemble = VectorAssembler(inputCols=cols, outputCol='features')
        sparkDFTrain = assemble.transform(sparkDFTrain)

        scale = StandardScaler(inputCol='features', outputCol='standardized')
        data_scale = scale.fit(sparkDFTrain)
        sparkDFTrain = data_scale.transform(sparkDFTrain)

        cols = sparkDFTest.columns.copy()
        cols.remove("label")
        assemble = VectorAssembler(inputCols=cols, outputCol='features')
        sparkDFTest = assemble.transform(sparkDFTest)

        scale = StandardScaler(inputCol='features', outputCol='standardized', )
        data_scale = scale.fit(sparkDFTest)
        sparkDFTest = data_scale.transform(sparkDFTest)

        nb = NaiveBayes(featuresCol='standardized', smoothing=1.0)
        nb = nb.fit(sparkDFTrain)
        pred = nb.transform(sparkDFTest)
        pred.show(10)

        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
        acc = evaluator.evaluate(pred)

        print("Prediction Accuracy: ", acc)  # Prediction Accuracy:  0.8908897080979397

        self.save_model(pred.toPandas())

    def save_model(self, df: pd.DataFrame):
        """Saves clustered data"""
        self.datamart.save_classified_data(df, "NB")


if __name__ == "__main__":
    conf = SparkConf() \
        .setAppName("MLE lab 3") \
        .setMaster("local")

    sc = SparkContext(conf=conf).getOrCreate()

    nbm = NaiveBayesModel()
    nbm.train()

    sc.stop()
