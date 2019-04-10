name := "house_price"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.4.0"

resolvers ++= Seq(
  "MMLSpark Repo" at "https://mmlspark.azureedge.net/maven",
  "apache-snapshots" at "http://repository.apache.org/snapshots/"
)

libraryDependencies ++= Seq(
  "com.microsoft.ml.spark" %% "mmlspark" % "0.16",
  "ml.dmlc" % "xgboost4j-spark" % "0.82",
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-hive" % sparkVersion
)
