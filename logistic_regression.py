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

        lr = LogisticRegression(featuresCol="standardized").fit(sparkDFTrain)

        lr_pred = lr.transform(sparkDFTest)

        preds = lr_pred.select('label', 'probability') \
            .rdd.map(lambda row: (float(row['probability'][1]), float(row['label']))) \
            .collect()

        summary = lr.summary

        summary.roc.show()
        print("areaUnderROC: " + str(summary.areaUnderROC))  # areaUnderROC: 1.0

        print(f'FP: {summary.falsePositiveRateByLabel}')  # FP: [0.0, 0.0]
        print(f'TP: {summary.truePositiveRateByLabel}')  # TP: [1.0, 1.0]

        fMeasure = summary.fMeasureByThreshold
        maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
        bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
            .select('threshold').head()['threshold']

        print(f"bestThreshold: {bestThreshold}")  # bestThreshold: 0.9995774624295405

        # from sklearn.metrics import roc_curve
        #
        # y_score, y_true = zip(*preds)
        # fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        #
        # print(f"FPR: {fpr}, TPR: {tpr}, Thresholds{thresholds}")

        # FPR: [0.         0.08658718 0.08725782 1.        ],
        # TPR: [0.         0.99924012 0.99924012 1.        ],
        # Thresholds[2.00000000e+00 1.00000000e+00 2.22044605e-16 0.00000000e+00]

        self.save_model(lr_pred.toPandas())

    def save_model(self, df: pd.DataFrame):
        """Saves clustered data"""
        self.datamart.save_classified_data(df, "LR")


if __name__ == "__main__":
    conf = SparkConf() \
        .setAppName("MLE lab 3") \
        .setMaster("local")

    sc = SparkContext(conf=conf).getOrCreate()

    nbm = LogisticRegressionModel()
    nbm.train()

    sc.stop()
