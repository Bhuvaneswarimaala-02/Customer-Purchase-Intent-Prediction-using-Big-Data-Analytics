from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler, PCA
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, ClusteringEvaluator
from pyspark.ml import Pipeline
import os

# Initialize Spark session with HADOOP_HOME set to avoid issues
spark = SparkSession.builder \
    .appName("ShoppersPurchaseIntent") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .getOrCreate()

# Load dataset into Spark DataFrame
file_path = r""
df_spark = spark.read.csv(file_path, header=True, inferSchema=True)

# Convert Boolean columns to Integer (Weekend and Revenue columns)
df_spark = df_spark.withColumn("Weekend", col("Weekend").cast("integer"))
df_spark = df_spark.withColumn("Revenue", col("Revenue").cast("integer"))

# Convert categorical columns to numerical using StringIndexer
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df_spark) 
            for column in ["Month", "VisitorType"]]

# Assemble features into a single vector
assembler = VectorAssembler(
    inputCols=['Administrative', 'Administrative_Duration', 'Informational', 
               'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
               'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
               'OperatingSystems', 'Browser', 'Region', 'TrafficType', 
               'Month_index', 'VisitorType_index', 'Weekend'],
    outputCol='features')

# Apply PCA to reduce dimensionality
pca = PCA(k=5, inputCol="features", outputCol="pcaFeatures")  # Set k=5 to reduce to 5 principal components

# Split the data into training and test sets
train_data, test_data = df_spark.randomSplit([0.7, 0.3], seed=42)

# Initialize evaluator for accuracy
classification_evaluator = MulticlassClassificationEvaluator(labelCol="Revenue", predictionCol="prediction", metricName="accuracy")
clustering_evaluator = ClusteringEvaluator(predictionCol="prediction", metricName="silhouette")

# Define classifiers (supervised)
classifiers = {
    "Logistic Regression": LogisticRegression(labelCol="Revenue", featuresCol="pcaFeatures", maxIter=100),
    "Naive Bayes": NaiveBayes(labelCol="Revenue", featuresCol="features")  # Naive Bayes doesn't use PCA features
}

# Define clustering algorithms (unsupervised)
clusterers = {
    "KMeans": KMeans(featuresCol="pcaFeatures", k=2),  # k=2 for binary clustering (buyer vs non-buyer)
    "Gaussian Mixture": GaussianMixture(featuresCol="pcaFeatures", k=2)
}

# Train and evaluate supervised models
for model_name, classifier in classifiers.items():
    print(f"\nTraining {model_name}...")

    # Set up the pipeline with PCA for supervised models
    stages = indexers + [assembler, pca, classifier] if model_name != "Naive Bayes" else indexers + [assembler, classifier]
    pipeline = Pipeline(stages=stages)

    # Train the model
    model = pipeline.fit(train_data)

    # Make predictions
    predictions = model.transform(test_data)

    # Evaluate the model
    accuracy = classification_evaluator.evaluate(predictions)
    print(f"{model_name} Accuracy: {accuracy:.4f}")

    # Save the model
    model_save_path = f"E:/BDA_MicroProject/{model_name}_model"
    # Ensure the directory exists
    os.makedirs(model_save_path, exist_ok=True)

    # Save the model
    model.write().overwrite().save(model_save_path)

# Train and evaluate unsupervised models
for model_name, clusterer in clusterers.items():
    print(f"\nTraining {model_name}...")

    # Set up the pipeline for clustering
    pipeline = Pipeline(stages=indexers + [assembler, pca, clusterer])

    # Train the model
    model = pipeline.fit(train_data)

    # Make predictions
    predictions = model.transform(test_data)

    # Rename the prediction column for clustering
    predictions = predictions.withColumnRenamed("prediction", f"{model_name}_prediction")

    # Evaluate the clustering model using silhouette score
    silhouette_score = clustering_evaluator.evaluate(predictions)
    print(f"{model_name} Silhouette Score: {silhouette_score:.4f}")

    # Save the model
    model_save_path = f"{model_name}_model"
    # Ensure the directory exists
    os.makedirs(model_save_path, exist_ok=True)

    # Save the model
    model.write().overwrite().save(model_save_path)

# Stop the Spark session
spark.stop()