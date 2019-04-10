import java.io.{BufferedWriter, FileWriter}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import scala.math.log1p
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, StringType}
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import com.microsoft.ml.spark.LightGBMRegressor
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.evaluation.RegressionEvaluator

object HousePrice {

  def main(args: Array[String]): Unit = {

    val debug = true

    /** Create SparkSession */

    val spark = SparkSession
      .builder()
      .master("localhost")
      .appName("House Price Predictor")
      .getOrCreate()

    if (debug) println("Connected to Spark")

    //import spark.implicits._

    // Display only ERROR logs in terminal

    spark.sparkContext.setLogLevel("ERROR")

    /** Get data */

    // Get current time

    //val xt = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYMMddHHmmss"))

    // Specify data files

    val trainData = "data/train.csv"
    val testData = "data/test.csv"

    // Specify output file

    //val outFile = new BufferedWriter(new FileWriter(("prediction_" + xt + ".txt")))

    // Create Train and Validation DataFrame using the train data file

    val trainSet = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",")
      .csv(trainData)
    //.na.drop() // remove rows with null or NaN values

    var Array(train, vali) = trainSet.randomSplit(Array(0.8, 0.2))

    // Create Test DataFrames using the test data file

    var test = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",")
      .csv(testData)
    //.na.drop() // remove rows with null or NaN values

    /** Prepare Features */

    //MSSubClass

    train = train
      .withColumn("MSSubClassTemp", $"MSSubClass".cast(StringType))
      .drop("MSSubClass")
      .withColumnRenamed("MSSubClassTemp", "MSSubClass")

    vali = vali
      .withColumn("MSSubClassTemp", $"MSSubClass".cast(StringType))
      .drop("MSSubClass")
      .withColumnRenamed("MSSubClassTemp", "MSSubClass")

    test = test
      .withColumn("MSSubClassTemp", $"MSSubClass".cast(StringType))
      .drop("MSSubClass")
      .withColumnRenamed("MSSubClassTemp", "MSSubClass")

    //MSZoning

    train = train.na.replace("MSZoning", Map[String, String]("C (all)" -> "C"))

    vali = vali.na.replace("MSZoning", Map[String, String]("C (all)" -> "C"))

    test = test.na.replace("MSZoning", Map[String, String]("C (all)" -> "C"))

    // LotFrontage
    train = train
      .na
      .replace("LotFrontage", Map[String, String]("NA" -> "-9999")) //to avoid div by 0 error
      .withColumn("LotFrontageTemp", $"LotFrontage".cast(DoubleType))
      .drop("LotFrontage")
      .withColumnRenamed("LotFrontageTemp", "LotFrontage")

    vali = vali
      .na
      .replace("LotFrontage", Map[String, String]("NA" -> "-9999")) //to avoid div by 0 error
      .withColumn("LotFrontageTemp", $"LotFrontage".cast(DoubleType))
      .drop("LotFrontage")
      .withColumnRenamed("LotFrontageTemp", "LotFrontage")

    test = test
      .na
      .replace("LotFrontage", Map[String, String]("NA" -> "-9999"))
      .withColumn("LotFrontageTemp", $"LotFrontage".cast(DoubleType))
      .drop("LotFrontage")
      .withColumnRenamed("LotFrontageTemp", "LotFrontage")

    // MasVnrType
    train = train
      .na
      .replace("MasVnrType", Map[String, String]("NA" -> null))

    vali = vali
      .na
      .replace("MasVnrType", Map[String, String]("NA" -> null))

    test = test
      .na
      .replace("MasVnrType", Map[String, String]("NA" -> null))

    // MasVnrArea
    train = train
      .na
      .replace("MasVnrArea", Map[String, String]("NA" -> "-9999"))
      .withColumn("MasVnrAreaTemp", $"MasVnrArea".cast(DoubleType))
      .drop("MasVnrArea")
      .withColumnRenamed("MasVnrAreaTemp", "MasVnrArea")

    vali = vali
      .na
      .replace("MasVnrArea", Map[String, String]("NA" -> "-9999"))
      .withColumn("MasVnrAreaTemp", $"MasVnrArea".cast(DoubleType))
      .drop("MasVnrArea")
      .withColumnRenamed("MasVnrAreaTemp", "MasVnrArea")

    test = test
      .na
      .replace("MasVnrArea", Map[String, String]("NA" -> "-9999"))
      .withColumn("MasVnrAreaTemp", $"MasVnrArea".cast(DoubleType))
      .drop("MasVnrArea")
      .withColumnRenamed("MasVnrAreaTemp", "MasVnrArea")

    //GarageYrBlt

    train = train
      .na
      .replace("GarageYrBlt", Map[String, String]("NA" -> "-9999"))
      .withColumn("GarageYrBltTemp", $"GarageYrBlt".cast(DoubleType))
      .drop("GarageYrBlt")
      .withColumnRenamed("GarageYrBltTemp", "GarageYrBlt")

    vali = vali
      .na
      .replace("GarageYrBlt", Map[String, String]("NA" -> "-9999"))
      .withColumn("GarageYrBltTemp", $"GarageYrBlt".cast(DoubleType))
      .drop("GarageYrBlt")
      .withColumnRenamed("GarageYrBltTemp", "GarageYrBlt")

    test = test
      .na
      .replace("GarageYrBlt", Map[String, String]("NA" -> "-9999"))
      .withColumn("GarageYrBltTemp", $"GarageYrBlt".cast(DoubleType))
      .drop("GarageYrBlt")
      .withColumnRenamed("GarageYrBltTemp", "GarageYrBlt")

    //For numeric Features, use -9999 for blank entries

    val numFeatures =
      for (tuple <- train.dtypes if !tuple._1.equals("SalePrice") && !tuple._1.equals("Id") && !tuple._2.equals("StringType"))
        yield tuple._1

    train = train
      .na
      .fill(-9999, numFeatures)

    vali = vali
      .na
      .fill(-9999, numFeatures)

    test = test
      .na
      .fill(-9999, numFeatures)

    // For non-numeric Features, index them

    val nonnumFeatures =
      for (tuple <- train.dtypes if !tuple._1.equals("SalePrice") && !tuple._1.equals("Id") && tuple._2.equals("StringType"))
        yield tuple._1

    for (col <- nonnumFeatures) {

      val stringIndexer: StringIndexer = new StringIndexer()
        .setInputCol(col)
        .setOutputCol(col + "_Indexed")
        .setHandleInvalid("keep") // to add new indexes for new labels

      val stringIndexerModel: StringIndexerModel = stringIndexer.fit(train)

      train = stringIndexerModel.transform(train)
      train = train.drop(col)

      vali = stringIndexerModel.transform(train)
      vali = train.drop(col)

      test = stringIndexerModel.transform(test)
      test = test.drop(col)
    }


    /** Prepare Label (Log (1 + Per SqFt price)) */

    train = train
      .withColumn("label", $"SalePrice")
      //.withColumn("perSqFt", $"SalePrice" / $"GrLivArea")
      //.withColumn("label", log1p($"perSqFt"))
      //.drop("perSqFt")

    vali = vali
      .withColumn("label", $"SalePrice")
    //.withColumn("perSqFt", $"SalePrice" / $"GrLivArea")
    //.withColumn("label", log1p($"perSqFt"))
    //.drop("perSqFt")

    /** Feature Transformation */

    // Combine the feature columns into a single vector column

    val featureColumns =
      for (col <- train.columns if !col.equals("SalePrice") && !col.equals("Id") && !col.equals("label"))
        yield col

    val vectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    train = vectorAssembler.transform(train)

    vali = vectorAssembler.transform(vali)

    /** Prepare Regressor */

    /*
    val regressor = new XGBoostRegressor()
      .setFeaturesCol("trainFeatures")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMaxBins(9999)
      .setMissing(-9999)
      .setSeed(7)
    */

    val regressor = new LightGBMRegressor()
      .setFeaturesCol(vectorAssembler.getOutputCol)
      .setLabelCol("label")
      .setPredictionCol("prediction")

    /** Train Model */

    val model = regressor.fit(train)

    /** Validate model with validation set*/

    val result = model.transform(vali)
    result.select("label", "prediction")

    /** Evaluate model */
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(result)

    println("RMSE: " + "%6.3f".format(rmse))

    spark.stop()
    if (debug) println("Disconnected from Spark")

  }

}
