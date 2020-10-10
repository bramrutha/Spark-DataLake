import configparser
import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['CREDENTIALS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['CREDENTIALS']['AWS_SECRET_ACCESS_KEY']

def create_spark_session():
    spark = SparkSession \
            .builder \
            .config("spark.jars.packages","org.apache.hadoop:hadoop-aws:2.7.0") \
            .getOrCreate()
    return spark


def process_song_data(spark,input_data,output_data):
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')

    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select('song_id','title','artist_id','year','duration').dropDuplicates()

    songs_table.printSchema()

    #write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(output_data + "songs/", mode="overwrite", partitionBy=["year","artist_id"])

    #extract columns to create artists
    artists_table = df.select('artist_id','artist_name','artist_location',
                              'artist_latitude','artist_longitude').dropDuplicates()

    #write artists table to parquet files
    artists_table.write.parquet(output_data + "artists/", mode="overwrite")


def process_log_data(spark,input_data,output_data):
    # get filepath to log data file
    log_data = os.path.join(input_data,'log_data/*/*/*.json')

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")

    # extract columns for users table
    users_table = df.select('userId','firstName','lastName','gender','level').dropDuplicates()

    #write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'users/'),mode = 'overwrite')

    # create timestamp column from original timestamp column
    get_timestamp = F.udf(lambda x : datetime.utcfromtimestamp(int(x)/1000),TimestampType())
    df = df.withColumn('start_time',get_timestamp("ts"))
    df = df.withColumn('hour',F.hour('start_time'))        \
           .withColumn('day', F.dayofmonth('start_time'))  \
           .withColumn('week', F.weekofyear('start_time')) \
           .withColumn('month', F.month('start_time'))     \
           .withColumn('year1', F.year('start_time'))       \
           .withColumn('weekday', F.dayofweek('start_time'))

    # extract columns to create time table
    time_table = df.select('start_time','hour','day','week','month','year1','weekday').dropDuplicates()

    #write time table to parquet files partitioned by year and month
    time_table.write.parquet(os.path.join(output_data, "time/"), mode='overwrite', partitionBy=["year1","month"])

    # read in song data to use for songplays table
    song_df = spark.read \
        .format("parquet") \
        .option("basePath", os.path.join(output_data, "songs/")) \
        .load(os.path.join(output_data, "songs/*/*/"))

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = df.join(song_df,df.song == song_df.title,how='inner')     \
                        .select(F.monotonically_increasing_id().alias('songplay_id'),
                                F.col('start_time'),
                                F.col('userId').alias('user_id'),
                                'level',
                                'song_id',
                                'artist_id',
                                F.col('sessionId').alias('session_id'),
                                'location',
                                F.col('userAgent').alias('user_agent'),
                                F.col('year1'),
                                F.col('month'))

    #write songplays table to parquet files partitioned by year and month
    songplays_table.drop_duplicates().write.parquet(os.path.join(output_data, "songplays/"), mode="overwrite", partitionBy=["year1","month"])



def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3://udacity-spark-datalake/output/"

    process_song_data(spark,input_data,output_data)
    process_log_data(spark,input_data,output_data)

if __name__ == "__main__":
    main()
