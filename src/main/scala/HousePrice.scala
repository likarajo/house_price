import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

object HousePrice {

  def main(args: Array[String]): Unit = {

    val debug = true

    val decisionTree = true;

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

    if (debug) println("Replaced NA values with 0")

    /** Prepare Features */

    val featureColumns = Array("MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MoSold", "YrSold")
    val labelColumn = "SalePrice"

    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val featurizedTrainingData = assembler.transform(trainingDF).select("Id", "features", "SalePrice")
    val featurizedTestData = assembler.transform(testDF).select("Id", "features")

    val Array(trainingData, validationData) = featurizedTrainingData.randomSplit(Array(0.8, 0.2))
    //val trainingData = featurizedTrainingData
    val testData = featurizedTestData

    if (debug) println("Features prepared")

    /** Choose regressor */

    val dt = new DecisionTreeRegressor()
      .setLabelCol("SalePrice")
      .setFeaturesCol("features")

    val rf = new RandomForestRegressor()
      .setLabelCol("SalePrice")
      .setFeaturesCol("features")

    val gbt = new GBTRegressor()
      .setLabelCol("SalePrice")
      .setFeaturesCol("features")
      .setMaxIter(10)

    if (debug) println("Regressors selected")

    /** Create Pipeline */

    val dtPipeline = new Pipeline()
      .setStages(Array(dt))

    val rfPipeline = new Pipeline()
      .setStages(Array(rf))

    val gbtPipeline = new Pipeline()
      .setStages(Array(gbt))

    if (debug) println("Pipeline created")

    /** Train and Prepare model */

    val dtModel = dtPipeline.fit(trainingData)

    if (debug) println("Learned Decision tree model")

    //println (dtModel.stages(0).asInstanceOf[DecisionTreeRegressionModel].toDebugString)

    val rfModel = rfPipeline.fit(trainingData)

    if (debug) println("Learned Random forest model")

    //println(rfModel.stages(0).asInstanceOf[RandomForestRegressionModel].toDebugString)

    val gbtModel = gbtPipeline.fit(trainingData)

    if (debug) println("Learned Gradient Boosting Tree model")

    // println(gbtModel.stages(0).asInstanceOf[GBTRegressionModel].toDebugString)

    /** Predict using model */

    val dtPredictions = dtModel.transform(validationData)

    if (debug) {
      println("Decision Tree Predictions")
      dtPredictions.select("id","SalePrice", "prediction").show(5)
    }

    val rfPredictions = rfModel.transform(validationData)

    if (debug) {
      println("Random Forest Predictions")
      rfPredictions.select("id","SalePrice", "prediction").show(5)
    }

    val gbtPredictions = gbtModel.transform(validationData)

    if (debug) {
      println("Gradient Boosting Tree Predictions")
      gbtPredictions.select("id","SalePrice", "prediction").show(5)
    }

    /** Evaluate Model */

    val regressionEvaluator = new RegressionEvaluator()
      .setLabelCol("SalePrice")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val dtRmse = regressionEvaluator.evaluate(dtPredictions)
    println("RMSE of Decision Tree model: " + dtRmse)

    val rfRmse = regressionEvaluator.evaluate(rfPredictions)
    println("RMSE of Random Forest model: " + rfRmse)

    val gbtRmse = regressionEvaluator.evaluate(gbtPredictions)
    println("RMSE of Gradient Boosting Tree model: " + gbtRmse)

    /*
    val outputFile = System.getProperty("user.dir") + "/housing-predictions/" + Calendar.getInstance()
      .getTime
      .toString
      .replaceAll(" ", "")

    dtPredictions.select("id", "prediction")
      .coalesce(1)
      .write
      .option("header", "true")
      .csv(outputFile)

    println("Decision Tree Prediction output exported as " + outputFile + ".csv")
    */

    spark.stop()

    if (debug) println("Disconnected from Spark")

  }

  def trainingSchema: StructType = StructType(dataSchema)

  def testSchema: StructType = StructType(dataSchema.dropRight(1)) // exclude label column

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

}
