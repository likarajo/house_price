import java.util.Calendar

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{LinearRegression, _}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

object HousePrice {

  def main(args: Array[String]): Unit = {

    if (args.length < 2) {
      System.err.println("Usage: HousePrice <input dir path> <output dir path>")
      System.exit(1)
    }

    val Array(inputDir, outputDir) = args.take(2)

    val time = Calendar.getInstance().getTime.toString.replaceAll(" ", "")
    //val writer = new PrintWriter(new File(outputDir + "/" + time + ".txt"))

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val debug = true

    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("House Price Prediction")
      .getOrCreate()
    if (debug) println("Connected to Spark")

    spark.sparkContext.setLogLevel("ERROR")

    /** Get Data */

    var trainingDF = spark.read
      .format("com.databricks.spark.csv") // allows reading CSV files as Spark DataFrames
      .option("delimiter", ",")
      .option("header", "true")
      .option("nullValue", "")
      .schema(trainingSchema)
      .load(inputDir + "/train.csv")
      .withColumnRenamed("SalePrice", "label")

    var testDF = spark.read
      .format("com.databricks.spark.csv") // allows reading CSV files as Spark DataFrames
      .option("delimiter", ",")
      .option("header", "true")
      .option("nullValue", "") // replace null values with NA
      .schema(testSchema)
      .load(inputDir + "/test.csv")

    if (debug) println("Data read into DataFrames")

    /** Pre-Processing Features */

    // Dropped Id column as it does not add any information
    trainingDF = trainingDF.drop("Id")
    testDF = testDF.drop("Id")

    val numericColumns =
      for (tuple <- trainingDF.dtypes if ! tuple._1.equals("label") && ! tuple._1.equals("Id") && ! tuple._2.equals("StringType")) yield tuple._1

    trainingDF = trainingDF.na.fill(-9999, numericColumns)
    testDF = testDF.na.fill(-9999, numericColumns)

    val categoricalColumns =
      for (tuple <- trainingDF.dtypes if ! tuple._1.equals("label") && ! tuple._1.equals("Id") && tuple._2.equals("StringType")) yield tuple._1

    // encode categorical feature
    for (col <- categoricalColumns) {
      val stringIndexer: StringIndexer = new StringIndexer()
        .setInputCol(col)
        .setOutputCol(col + "_Index")
        .setHandleInvalid("keep")
      val stringIndexerModel: StringIndexerModel = stringIndexer.fit(trainingDF)
      trainingDF = stringIndexerModel.transform(trainingDF)
      testDF = stringIndexerModel.transform(testDF)
      trainingDF = trainingDF.drop(col)
      testDF = testDF.drop(col)
    }

    val featureColumns =
      for (col <- trainingDF.columns if ! col.equals("label") && ! col.equals("Id")) yield col

    val vectorAssembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")
    trainingDF = vectorAssembler.transform(trainingDF).select("Id", "features", "label")
    testDF = vectorAssembler.transform(testDF).select("Id", "features")

    if (debug) trainingDF.show(2)
    if (debug) testDF.show(2)

    /** Choose Regressors */

    val lr = new LinearRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setImpurity("variance")

    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setImpurity("variance")

    val gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setImpurity("variance")

    if (debug) println("Regressors selected")

    /** Create Pipeline and Parameter builder for Hyper-parameter tuning of models */

    val pipeline = new Pipeline()

    val paramGrid_lr = new ParamGridBuilder()
      .baseOn(pipeline.stages -> Array[PipelineStage](lr))
      .addGrid(lr.maxIter, Array(10, 20, 30))
      .addGrid(lr.regParam, Array(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.1))
      .addGrid(lr.elasticNetParam, Array(0.7, 0.8, 0.9))
      .build()

    val paramGrid_dt = new ParamGridBuilder()
      .baseOn(pipeline.stages -> Array[PipelineStage](dt))
      .addGrid(dt.maxDepth, Array(2, 3, 5, 10))
      .build()

    val paramGrid_rf = new ParamGridBuilder()
      .baseOn(pipeline.stages -> Array[PipelineStage](rf))
      .addGrid(rf.maxDepth, Array(2, 3, 5, 10))
      .addGrid(rf.numTrees, Array(5, 10, 15, 20))
      .build()

    val paramGrid_gbt = new ParamGridBuilder()
      .baseOn(pipeline.stages -> Array[PipelineStage](gbt))
      .addGrid(gbt.maxDepth, Array(2, 3, 5, 10))
      .addGrid(gbt.maxIter, Array(5, 10, 15, 20))
      .build()

    val paramGrid = paramGrid_lr //++ paramGrid_dt ++ paramGrid_rf ++ paramGrid_gbt

    if (debug) println("Pipeline and Parameter grid built for models")

    /** Set evaluator */

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    /** Find best model */

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setParallelism(3) // Evaluate up to 3 parameter settings in parallel

    if (debug) println("Cross validator set for finding best model parameters")

    /** Split the data into training and test sets */

    val Array(training, validation) = trainingDF.randomSplit(Array(0.9, 0.2), seed = 11L)

    if (debug) println("Training and Validation sets formed")

    /** Training with Best Model */

    println("Running cross-validation to choose the best model...")

    val cvModel = cv.fit(training)

    println("Best model found")

    val bestModel = cvModel.bestEstimatorParamMap
    println(bestModel)

    /** Make Prediction on validation set using model */

    val prediction = cvModel.transform(validation)

    println("Predictions made on validation set\n")

    /** Evaluate Model */

    val rmse = evaluator.evaluate(prediction)
    println(s"RMSE of Best Model is: $rmse")

    spark.stop()

    if (debug) println("Disconnected from Spark")

  }

  implicit class BestParamMapCrossValidatorModel(cvModel: CrossValidatorModel) {
    def bestEstimatorParamMap: ParamMap = {
      cvModel.getEstimatorParamMaps
        .zip(cvModel.avgMetrics)
        .maxBy(_._2)
        ._1
    }
  }

  def trainingSchema: StructType = StructType(dataSchema)

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

  def testSchema: StructType = StructType(dataSchema.dropRight(1)) // exclude label column

}
