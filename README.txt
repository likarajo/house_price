House Price Prediction

Steps to run:

1) The training and test data are in data/ folder:
    data/train.csv
    data/test.csv

    These files are needed to be uploaded to S3:
    s3://bucket/data/train.csv
    s3://bucket/data/test.csv


2) The program (or jar) file also needs to be uploaded to S3:
    s3://bucket/house_price_2.11-0.1.jar

3) The program needs to be run on a cluster having Spark 2.4.0 or a spark shell
    The arguments that are needed to be passed to the program are:
    training file path
    test file path
    output directory path

    Syntax:

    spark-submit --class HousePrice s3://bucket/house_price_2.11-0.1.jar s3://bucket/data/train.csv s3://bucket/data/test.csv s3://bucket

4) The result is obtained in a file that gets created in the output directory specified.

5) Sample output

Best Model:
{
	linReg_610dcd27429d-elasticNetParam: 0.7,
	linReg_610dcd27429d-maxIter: 10,
	linReg_610dcd27429d-regParam: 0.5,
	pipeline_9901a92912ea-stages: [Lorg.apache.spark.ml.PipelineStage;@16c58781
}
RMSE of Best Model on validation set is: 0.129102130712155
Predictions on test data
+----+------------------+
|  Id|PredictedSalePrice|
+----+------------------+
|1461|112092.32601502787|
|1462|158632.97458251583|
|1463|176595.49976749351|
|1464|199689.07345033245|
|1465|193424.50447121722|
|1466|171593.45264224903|
|1467|188041.81298588397|
+----+------------------+
