# Importing all the required Packages
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.sql.functions import dayofweek, year, month
from awsglue.dynamicframe import DynamicFrame

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

## Read the data 
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "test_ v", table_name = "input", transformation_ctx = "datasource0")

# Convert the DynamicFrame to Spark DataFrame
df_data = datasource0.toDF()

# from date field we are extracting "day_of_week", "month" and year
df_data_1=df_data.withColumn('day_of_week',dayofweek(df_data.date)).withColumn('month', month(df_data.date)).withColumn("year", year(df_data.date))

# Droping date coloum
df_data_2=df_data_1.drop('date')

# Remove the target column from the input feature set.
featuresCols = df_data_2.columns
featuresCols.remove('sales')
 
# vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")

data_vector = vectorAssembler.transform(df_data_2)

df_data_2=df_data_2.withColumnRenamed("sales", "label")

data =df_data_2.select("label", "rawFeatures")

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a GBT model.
gbt = GBTRegressor(featuresCol="features", maxIter=15)

# Chain indexer and GBT in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, gbt])

# Train model.  This also runs the indexer.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

gbtModel = model.stages[1]

# Convert datafrmae back to a Dynamic DataFrame
test_nest=DynamicFrame.fromDF(predictions, glueContext, "test_nest")

# Save the result back to S3 bucket
datasink2 = glueContext.write_dynamic_frame.from_options(frame = test_nest, connection_type = "s3", connection_options = {"path": "s3://glue-job-customscript/input"}, format = "csv", transformation_ctx = "datasink2")
job.commit()