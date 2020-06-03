from pyspark import SparkConf, SparkContext, SQLContext
import pyspark.sql.functions as functions
from pyspark.sql.types import *
import os
from pyspark.sql.window import Window
from pyspark import Broadcast

conf = SparkConf().set("spark.driver.host", "127.0.0.1").set("spark.sql.shuffle.partitions", "8")
sc = SparkContext(conf=conf, master="local[*]", appName="Pyspark-Practice")
sqlContext = SQLContext(sc)

folder_path = os.path.join(os.getcwd(), "input_data")

'''find top 2 salaried people in each department'''
def find_rank():
    input_data_path = os.path.join(folder_path, 'salary', 'salary.csv')
    df =sqlContext.read.option("header", "true") \
            .option("multiline", "true") \
            .csv(input_data_path)
    wind = Window.partitionBy("depName").orderBy(functions.col("salary").desc())
    winCol = functions.rank().over(wind)
    df.select("*", winCol.alias("rank")) \
        .filter(functions.col("rank") <= 2) \
        .show()
'''find average salary of each department'''
def find_average_salary():
    input_data_path = os.path.join(folder_path, 'salary', 'salary.csv')
    df = sqlContext.read.option("header", "true") \
            .option("multiline", "true") \
            .csv(input_data_path)
    wind = Window.partitionBy("depName")
    windCol = functions.round(functions.mean(functions.col("salary")).over(wind), 2)
    df.select("*", windCol.alias("average salary")) \
        .show()

'''Top N per Group. A dataset containing product, category and revenue is provided.
Find best-selling and the second best-selling products in every category'''
def find_top_n():
    input_data_path = os.path.join(folder_path, 'revenue', 'revenue.csv')
    df = sqlContext.read \
        .option("header", "true") \
        .option("multiline", "true") \
        .csv(input_data_path)
    wind = Window.partitionBy("category").orderBy("revenue")
    windCol = functions.dense_rank().over(wind)
    df.select("*", windCol.alias("rank")) \
        .filter(functions.col("rank") == 1) \
        .show()

'''given an orders table containing timestamp.Find the last order placed for each user'''
def find_time_lag():
    input_data_path = os.path.join(folder_path, 'orders', 'orders.csv')
    df = sqlContext.read \
        .option("multiline", "true") \
        .option("header", "true") \
        .csv(input_data_path)
    wind = Window.partitionBy("userId").orderBy("timestamp")
    windCol = functions.col("timestamp") - functions.lag("timestamp").over(wind)
    df.select("*", functions.when(windCol.isNotNull(), windCol).otherwise(0) \
        .alias("previous_order_timestamp").cast(LongType())) \
        .show()

'''find the sum of all order quantity including the current one'''
def running_total():
    input_data_path = os.path.join(folder_path, 'orders_data', 'orders.csv')
    df = sqlContext.read \
        .option("multiline", "true") \
        .option("header", "true") \
        .csv(input_data_path)
    wind = Window.orderBy("id")
    windCol = functions.sum("orderQty").over(wind)
    df.select("*", windCol.alias("totalQuantity").cast(IntegerType())) \
        .show()



'''given a text file, find most commong words in it, exclude the stopwords'''
def word_count():
    input_data_path = os.path.join(folder_path, 'wordcount', 'testfile.txt')
    stopwords_data_path = os.path.join(folder_path, 'wordcount', 'exclude.txt')
    with open(stopwords_data_path) as f:
        my_list = list(f)
        stopwords = list(map(lambda word : word.strip(), my_list))
    df = sqlContext.read \
        .text(input_data_path)
    df.withColumn("value_lower", functions.lower(functions.col("value"))) \
        .withColumn("value_list", functions.split(functions.col("value_lower"), " ")) \
        .drop("value") \
        .drop("value_lower") \
        .withColumn("value_list_ex", functions.explode(functions.col("value_list"))) \
        .filter(~ functions.col("value_list_ex").isin(stopwords)) \
        .groupBy("value_list_ex").count().orderBy(functions.col("count").desc()) \
        .show()

if __name__ == "__main__":
    find_rank()
    find_average_salary()
    find_top_n()
    find_time_lag()
    running_total()
    word_count()