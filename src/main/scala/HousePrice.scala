import java.util.Calendar

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

object HousePrice {

  def dataSchema = Array(
    StructField("Id", IntegerType, true),
    StructField("MSSubClass", IntegerType, true),
    StructField("MSZoning", StringType, true),
    StructField("LotFrontage", IntegerType, true),
    StructField("LotArea", IntegerType, true),
    StructField("Street", StringType, true),
    StructField("Alley", StringType, true),
    StructField("LotShape", StringType, true),
    StructField("LandContour", StringType, true),
    StructField("Utilities", StringType, true),
    StructField("LotConfig", StringType, true),
    StructField("LandSlope", StringType, true),
    StructField("Neighborhood", StringType, true),
    StructField("Condition1", StringType, true),
    StructField("Condition2", StringType, true),
    StructField("BldgType", StringType, true),
    StructField("HouseStyle", StringType, true),
    StructField("OverallQual", IntegerType, true),
    StructField("OverallCond", IntegerType, true),
    StructField("YearBuilt", IntegerType, true),
    StructField("YearRemodAdd", IntegerType, true),
    StructField("RoofStyle", StringType, true),
    StructField("RoofMatl", StringType, true),
    StructField("Exterior1st", StringType, true),
    StructField("Exterior2nd", StringType, true),
    StructField("MasVnrType", StringType, true),
    StructField("MasVnrArea", StringType, true),
    StructField("ExterQual", StringType, true),
    StructField("ExterCond", StringType, true),
    StructField("Foundation", StringType, true),
    StructField("BsmtQual", StringType, true),
    StructField("BsmtCond", StringType, true),
    StructField("BsmtExposure", StringType, true),
    StructField("BsmtFinType1", StringType, true),
    StructField("BsmtFinSF1", IntegerType, true),
    StructField("BsmtFinType2", StringType, true),
    StructField("BsmtFinSF2", IntegerType, true),
    StructField("BsmtUnfSF", IntegerType, true),
    StructField("TotalBsmtSF", IntegerType, true),
    StructField("Heating", StringType, true),
    StructField("HeatingQC", StringType, true),
    StructField("CentralAir", StringType, true),
    StructField("Electrical", StringType, true),
    StructField("1stFlrSF", IntegerType, true),
    StructField("2ndFlrSF", IntegerType, true),
    StructField("LowQualFinSF", IntegerType, true),
    StructField("GrLivArea", IntegerType, true),
    StructField("BsmtFullBath", IntegerType, true),
    StructField("BsmtHalfBath", IntegerType, true),
    StructField("FullBath", IntegerType, true),
    StructField("HalfBath", IntegerType, true),
    StructField("BedroomAbvGr", IntegerType, true),
    StructField("KitchenAbvGr", IntegerType, true),
    StructField("KitchenQual", StringType, true),
    StructField("TotRmsAbvGrd", IntegerType, true),
    StructField("Functional", StringType, true),
    StructField("Fireplaces", IntegerType, true),
    StructField("FireplaceQu", StringType, true),
    StructField("GarageType", StringType, true),
    StructField("GarageYrBlt", IntegerType, true),
    StructField("GarageFinish", StringType, true),
    StructField("GarageCars", IntegerType, true),
    StructField("GarageArea", IntegerType, true),
    StructField("GarageQual", StringType, true),
    StructField("GarageCond", StringType, true),
    StructField("PavedDrive", StringType, true),
    StructField("WoodDeckSF", IntegerType, true),
    StructField("OpenPorchSF", IntegerType, true),
    StructField("EnclosedPorch", IntegerType, true),
    StructField("3SsnPorch", IntegerType, true),
    StructField("ScreenPorch", IntegerType, true),
    StructField("PoolArea", IntegerType, true),
    StructField("PoolQC", StringType, true),
    StructField("Fence", StringType, true),
    StructField("MiscFeature", StringType, true),
    StructField("MiscVal", IntegerType, true),
    StructField("MoSold", IntegerType, true),
    StructField("YrSold", IntegerType, true),
    StructField("SaleType", StringType, true),
    StructField("SaleCondition", StringType, true),
    StructField("SalePrice", IntegerType, true)
  )

  def trainingSchema: StructType = StructType(dataSchema)

  def testSchema: StructType = StructType(dataSchema.dropRight(1)) // exclude label column

  def main(args: Array[String]): Unit = {

    val debug = true

    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("House Price Prediction")
      .getOrCreate()
    if (debug) println("Connected to Spark")

    spark.sparkContext.setLogLevel("ERROR")

    /** Get Data */

    val srcDataDir = System.getProperty("user.dir") + "/data/"

    var trainingDF = spark.read
      .format("com.databricks.spark.csv") // allows reading CSV files as Spark DataFrames
      .option("delimiter", ",")
      .option("header", "true")
      .option("nullValue", "NA") // replace null values with NA
      .schema(trainingSchema)
      .load(srcDataDir + "train.csv")

    var testDF = spark.read
      .format("com.databricks.spark.csv") // allows reading CSV files as Spark DataFrames
      .option("delimiter", ",")
      .option("header", "true")
      .option("nullValue", "NA") // replace null values with NA
      .schema(testSchema)
      .load(srcDataDir + "test.csv")

    if (debug) println("Data read into Dataframes")

    trainingDF = trainingDF.select("Id", "SalePrice", "MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MoSold", "YrSold")
      .na.fill(0) // replace NA with 0

    testDF = testDF.select("Id", "MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MoSold", "YrSold")
      .na.fill(0) // replace NA with 0

    if (debug) println("Replaces NA values with 0")

    /** Prepare Features */

    val featureColumns = Array("MSSubClass","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MoSold","YrSold")
    val labelColumn = "SalePrice"

    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val featurizedTrainingData = assembler.transform(trainingDF).select("Id","features", "SalePrice")
    val featurizedTestData = assembler.transform(testDF).select("Id","features")

    // val Array(trainingData, validationData) = featurizedTrainingData.randomSplit(Array(0.8, 0.2))
    val trainingData = featurizedTrainingData
    val testData = featurizedTestData

    if (debug) println("Features prepared")

    /** Choose regressor */

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("SalePrice")
      .setFeaturesCol("features")

    if (debug) println("Regressor selected")

    /** Create Pipeline */

    val pipeline2 = new Pipeline()
      .setStages(Array(dt))

    if (debug) println("Pipeline created")

    /** Train and Prepare model */

    val model2 = pipeline2.fit(trainingData)

    if (debug) println("Model trained")

    /** Predict using model */

    val predictions2 = model2.transform(testData)

    // Select example rows to display.
    val outputFile = System.getProperty("user.dir") + "/housing-predictions/" + Calendar.getInstance()
      .getTime
      .toString
      .replaceAll(" ", "")

    predictions2.withColumnRenamed("prediction", "Predicted Sale Price")
      .select("Id","Predicted Sale Price")
      .coalesce(1)
      .write
      .option("header", "true")
      .csv(outputFile)

    predictions2.show(10)

    println("Prediction output exported as " + outputFile + ".csv")

    // Select (prediction, true label) and compute test error.
    //    val evaluator2 = new RegressionEvaluator()
    //      .setLabelCol("label")
    //      .setPredictionCol("prediction")
    //      .setMetricName("rmse")
    //    val rmse2 = evaluator2.evaluate(predictions2)
    //
    //    val treeModel = model2.stages(0).asInstanceOf[DecisionTreeRegressionModel]
    //    println("Learned regression tree model:\n" + treeModel.toDebugString)

    // Train a RandomForest model.
    //    val rf = new RandomForestRegressor()
    //      .setLabelCol("SalePrice")
    //      .setFeaturesCol("features")
    //
    //    // Chain indexer and forest in a Pipeline.
    //    val rfPipeline = new Pipeline()
    //      .setStages(Array(rf))
    //
    //    // Train model. This also runs the indexer.
    //    val rfFittedModel = rfPipeline.fit(trainingData)
    //
    //    // Make predictions.
    //    val rfPredictions = rfFittedModel.transform(testData)
    //
    //    // Select example rows to display.
    //    rfPredictions.select("prediction", "SalePrice", "features").show(5)
    //
    //    // Select (prediction, true label) and compute test error.
    //    val rfevaluator = new RegressionEvaluator()
    //      .setLabelCol("SalePrice")
    //      .setPredictionCol("prediction")
    //      .setMetricName("rmse")
    //    val RFrmse = rfevaluator.evaluate(rfPredictions)
    //
    //    val rfModel = rfFittedModel.stages(0).asInstanceOf[RandomForestRegressionModel]
    //    println("Learned regression forest model:\n" + rfModel.toDebugString)
    //
    //    // Train a GBT model.
    //    val gbt = new GBTRegressor()
    //      .setLabelCol("SalePrice")
    //      .setFeaturesCol("features")
    //      .setMaxIter(10)
    //
    //    // Chain indexer and GBT in a Pipeline.
    //    val pipeline = new Pipeline()
    //      .setStages(Array(gbt))
    //
    //    // Train model. This also runs the indexer.
    //    val model = pipeline.fit(trainingData)
    //
    //    // Make predictions.
    //    val predictions = model.transform(testData)
    //
    //    // Select example rows to display.
    //    predictions.select("prediction", "SalePrice", "features").show(5)
    //
    //    // Select (prediction, true label) and compute test error.
    //    val evaluator = new RegressionEvaluator()
    //      .setLabelCol("SalePrice")
    //      .setPredictionCol("prediction")
    //      .setMetricName("rmse")
    //    val rmse = evaluator.evaluate(predictions)
    //    println("Root Mean Squared Error (RMSE) on GBT model test data = " + rmse)
    //
    //    val gbtModel = model.stages(0).asInstanceOf[GBTRegressionModel]
    //
    //
    //
    //    println("Learned regression GBT model:\n" + gbtModel.toDebugString)
    //
    //    println("Decision Tree RMSE = " + rmse2)
    //    println("RandomForestRegressor RMSE = " + RFrmse)
    //    println("Gradient-boosted Tree RMSE = " + rmse)

    if (debug) println("Disconnected from Spark")

    spark.stop()

  }

}
