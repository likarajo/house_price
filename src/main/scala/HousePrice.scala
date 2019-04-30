import java.io.{File, PrintWriter}
import java.util.Calendar

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{LinearRegression, _}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.log1p

object HousePrice {

  def main(args: Array[String]): Unit = {

    if (args.length < 2) {
      System.err.println("Usage: HousePrice <input dir path> <output dir path>")
      System.exit(1)
    }

    val Array(inputDir, outputDir) = args.take(2)

    val time = Calendar.getInstance().getTime.toString.replaceAll(" ", "")
    val writer = new PrintWriter(new File(outputDir + "/" + time + "_model.txt"))

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
      .option("inferSchema", "true")
      .load(inputDir + "/train.csv")

    if (debug) println("Data read into DataFrames")

    /** Pre-Processing Features */

    // Drop unimportant columns
    trainingDF = trainingDF.drop("1stFlrSF", "GarageYrBlt", "GarageArea", "1stFloor", "TotRmsAbvGrd", "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "LotFrontage",
      "GarageCond", "GarageType", "GarageFinish", "GarageQual", "BsmtExposure", "BsmtFinType2", "BsmtFinType1", "BsmtCond", "BsmtQual", "MasVnrArea", "MasVnrType")

    // Drop null value rows
    trainingDF = trainingDF.filter(trainingDF("Electrical")=!="NA")
    trainingDF = trainingDF.filter(!trainingDF("Id").isin(1299,524))

    // Remove skewness
    trainingDF = trainingDF.withColumn("SalePrice", log1p("SalePrice"))
    trainingDF = trainingDF.withColumn("GrLivArea", log1p("GrLivArea"))
    trainingDF = trainingDF.withColumn("TotalBsmtSF", log1p("TotalBsmtSF"))

    val categoricalColumns =
      for (tuple <- trainingDF.dtypes if ! tuple._1.equals("SalePrice") && ! tuple._1.equals("Id") && tuple._2.equals("StringType")) yield tuple._1

    // encode categorical feature
    for (col <- categoricalColumns) {
      val stringIndexer: StringIndexer = new StringIndexer()
        .setInputCol(col)
        .setOutputCol(col + "_Index")
      val stringIndexerModel: StringIndexerModel = stringIndexer.fit(trainingDF)
      trainingDF = stringIndexerModel.transform(trainingDF)
      trainingDF = trainingDF.drop(col)
    }

    val featureColumns =
      for (col <- trainingDF.columns if ! col.equals("SalePrice") && ! col.equals("Id")) yield col

    val vectorAssembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    trainingDF = vectorAssembler.transform(trainingDF)

    trainingDF = trainingDF.select("Id", "features", "SalePrice")
      .toDF("Id", "features", "label")

    if (debug) trainingDF.show(2)

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
      .addGrid(lr.regParam, Array(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1))
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
      .addGrid(gbt.maxDepth, Array(1, 2, 3, 4, 5))
      .addGrid(gbt.maxIter, Array(1, 2, 3, 4, 5))
      .build()

    val paramGrid =   paramGrid_rf // ++ paramGrid_gbt ++ paramGrid_lr ++ paramGrid_dt

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

    val Array(training, validation) = trainingDF.randomSplit(Array(0.9, 0.2))

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
    println(s"RMSE of Best Model on validation set is: $rmse")

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

}
