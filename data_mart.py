import pandas as pd
from sklearn.utils import shuffle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from logger import Logger
import sys
import traceback

SHOW_LOG = True


class DataMart:

    NUM_CLASTERS = 2

    def __init__(self):
        """Reads data from data source, proceeds it and makes test, train split"""
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)

        self.sqlEngine = create_engine('mysql+pymysql://artem:artem@127.0.0.1:6603/MLE_LAB_3', pool_recycle=3600)

        conn = self.sqlEngine.connect()
        df = pd.read_sql("SELECT * FROM data_classified;", conn)
        df.drop(["index", "features", "standardized"], axis=1, inplace=True)
        conn.close()

        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()
        for i in range(DataMart.NUM_CLASTERS):
            df_class = df[df['prediction'] == i]

            if len(df_class) < 100:
                df.drop(df[df["prediction"] == i].index, inplace=True)
                continue

            df_train_class, df_test_class = train_test_split(df_class, test_size=0.2)

            if i == 0:
                self.df_train = df_train_class
                self.df_test = df_test_class
            else:
                self.df_train = pd.concat([self.df_train, df_train_class])
                self.df_test = pd.concat([self.df_test, df_test_class])

        self.df_train = shuffle(self.df_train)
        self.df_test = shuffle(self.df_test)

    def get_train_data(self):
        return self.df_train

    def get_test_data(self):
        return self.df_test

    def get_data_test_classified_by_nb(self):
        conn = self.sqlEngine.connect()
        df = pd.read_sql("SELECT * FROM data_test_set_classified_by_NB;", conn)
        df.drop(["index", "features", "standardized", "rawPrediction", "probability"], axis=1, inplace=True)
        conn.close()

        return df

    def save_classified_data(self, data: pd.DataFrame, method: str):
        conn = self.sqlEngine.connect()
        try:
            data.to_sql(f"data_test_set_classified_by_{method}", conn, if_exists="replace")
            self.df_train.to_sql("data_train_set", conn, if_exists="replace")
            self.df_test.to_sql("data_test_set", conn, if_exists="replace")
        except ValueError as e:
            self.log.error(e)
            sys.exit(1)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        else:
            self.log.info("Classisied data added to database successfully")
        finally:
            conn.close()

        conn.close()


if __name__ == "__main__":
    dm = DataMart()

    print(dm.get_train_data())
    print(dm.get_test_data())




