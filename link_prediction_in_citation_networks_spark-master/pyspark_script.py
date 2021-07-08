### Please set your spark context below ###
### Please set the correct file paths at lines 75 and 82 ####

import sys
import os
import csv
import time
import findspark
findspark.init('C:/Bigdata/spark') 
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkContext, SparkConf

spark = SparkSession.builder.master("local[2]").appName("link").getOrCreate()
sc = spark.sparkContext
spark.sparkContext.setLogLevel("ERROR")

from pyspark.sql.types import StructType, ArrayType, IntegerType, StructField, StringType
from pyspark.mllib.classification import LogisticRegressionWithSGD

# session
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
# utils
from pyspark.sql.functions import udf, struct, split, when
from pyspark.sql.types import IntegerType, FloatType, ArrayType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as F
# features
from pyspark.ml.feature import StopWordsRemover, VectorAssembler, VectorIndexer
from pyspark.ml.feature import IndexToString, StringIndexer, HashingTF, IDF
from pyspark.ml.feature import IDFModel, Tokenizer, CountVectorizer, Normalizer
# models
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


start = time.time()

# Utility to avoid operations on `None`s and empty lists
def type_check(func):
    def wrapper(list1, list2):
        if (not list1) or (not list2):
            return 0
        return func(list1, list2)
    return wrapper


@type_check
def findNumberTokens (list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2))

@type_check
def simJaccard (list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2))/(float(len(set1.union(set2))))

def union (list1,list2):
    return list(set(set(list1).union(set(list2))))

def array(x):
    if type(x) is not list:
        return []
    else:
        return x


### Please set the correct file paths ####
# Create a dataframe from training_set
trainingRDD = sc.textFile("C:/Users/saket/OneDrive/Desktop/Mini Project/link_prediction_in_citation_networks_spark-master/training_set.txt")
trainingRDD = trainingRDD.mapPartitions(lambda x: csv.reader(x, delimiter=" "))
trainingDF = trainingRDD.toDF(['from_node_id', 'to_node_id', 'label']).sample(False, 0.10, 10)

trainingDF.show(5)

# Create a dataframe for paper information (title, authors, abstract, etc)
infoRDD = sc.textFile("C:/Users/saket/OneDrive/Desktop/Mini Project/link_prediction_in_citation_networks_spark-master/node_information.csv")
infoRDD = infoRDD.mapPartitions(lambda x: csv.reader(x))
infoDF = infoRDD.toDF(['node_id', 'year', 'title', 'authors', 'journal', 'abstract'])
infoDF.printSchema()
infoDF.show(5)


## Join and rename _FROM part
newDF = infoDF.join(trainingDF, infoDF.node_id == trainingDF.from_node_id).select('from_node_id', 'title', 'year', 'authors', 'journal', 'abstract', 'to_node_id', 'label')

for col in ["title", "year", "authors", "abstract", "journal"]:
    newDF = newDF.withColumnRenamed(col, col+"_from")
    
    
## Join and rename _TO part
newDF = infoDF.join(newDF, infoDF.node_id == newDF.to_node_id).select('from_node_id', 'title_from', 'year_from', 'authors_from',
            'abstract_from', 'journal_from', 'to_node_id', 'title',
            'year', 'authors', 'journal', 'abstract', 'label', 
         )

for col in ["title", "year", "authors", "abstract", "journal"]:
    newDF = newDF.withColumnRenamed(col, col+"_to")


# Change the data type of specific columns.
newDF = newDF.withColumn('year_from', newDF["year_from"].cast(IntegerType()))
newDF = newDF.withColumn('year_to', newDF["year_to"].cast(IntegerType()))
newDF = newDF.withColumn('label', newDF['label'].cast(IntegerType()))


for col in ["title", "abstract", "journal"]:
    for src in ["from", "to"]:
        newDF = newDF.withColumn(col+'_'+src+'_words', split(col+"_"+src, "\s+"))
        remover = StopWordsRemover(inputCol=col+'_'+src+'_words', outputCol=col+'_'+src+'_words_f')
        newDF = remover.transform(newDF)

# Tokenize authors
newDF = newDF.withColumn('authors_from_words_f', split("authors_from", ", "))
newDF = newDF.withColumn('authors_to_words_f', split("authors_to", ", "))


udf_token_overlap = udf(lambda x: findNumberTokens(x[0], x[1]), returnType=IntegerType())
udf_abs = udf(lambda x: abs(x[0] - x[1]), returnType=IntegerType())
udf_simJaccard = udf(lambda x: simJaccard(x[0], x[1]), returnType=FloatType())
udf_union = udf(lambda x: union(x[0],x[1]), returnType=ArrayType(StringType()))
udf_add = udf(lambda x: (x[0] + x[1]), returnType=IntegerType())

#a udf to replace null values to empty arrays (not used in this script)
udf_array = udf(lambda x: array(x[0]), returnType=ArrayType(StringType()))

udf_tovector = udf(lambda x: Vectors.dense(x), returnType=VectorUDT())


# calculate features
for col in ["title", "authors", "abstract", "journal"]:
    # overlap
    newDF = eval("newDF.withColumn('"+col+"_overlap', udf_token_overlap(struct(newDF."+col+"_from_words_f, newDF."+col+"_to_words_f)))")
    newDF = eval("newDF.withColumn('"+col+"_simJaccard', udf_simJaccard(struct(newDF."+col+"_from_words_f, newDF."+col+"_to_words_f)))")
    

newDF = newDF.withColumn('time_dist', udf_abs(struct(newDF.year_from, newDF.year_to)))
newDF = newDF.withColumn('same_journal', when(newDF.journal_from == newDF.journal_to, 1).otherwise(0))





# Find number of outgoing edges per node
nodes = newDF.select("from_node_id", "to_node_id", "label")
nodes = nodes.filter(nodes.label == 1)

nodes.show(5)

#Create the outgoing links for all the nodes, we are creating 2 different DFs since there is problem with Spark and crossjoins.

mydf1 = nodes.groupBy("from_node_id").agg(F.count("to_node_id").alias('from_node_degree'),F.collect_set('to_node_id').alias('from_node_neighbors'))
mydf1 = mydf1.withColumnRenamed('from_node_id', 'node_id_from')

mydf2 = nodes.groupBy("from_node_id").agg(F.count("to_node_id").alias('to_node_degree'),F.collect_set('to_node_id').alias('to_node_neighbors'))
mydf2 = mydf2.withColumnRenamed('from_node_id', 'node_id_to')

#join everthing to the initial DF
left_join = newDF.join(mydf1, newDF.from_node_id == mydf1.node_id_from, how='left') # Could also use 'left_outer'
join = left_join.join(mydf2, left_join.to_node_id == mydf2.node_id_to, how='left') # Could also use 'left_outer'
newDF = join

#Here we are creating the number of common neighbors which stands for the number of common outgoing links
newDF = newDF.withColumn("n_common_neigbors", udf_token_overlap(struct(newDF.from_node_neighbors,
								       newDF.to_node_neighbors)))


newDF.printSchema()

# filling null values with 0 in order to find the absolute differnce 
newDF = newDF.na.fill(0)

#since we had the degree for each node we find the difference in the degree for each pair of our dataset
newDF = newDF.withColumn("diff_in_degree", udf_abs(struct(newDF.from_node_degree,
								       newDF.to_node_degree)))


# Find number of incoming edges for each node "who cites a node", again we repeat the process in order to avoid the crosjoin problem

mydf3 = nodes.groupBy("to_node_id").agg(F.count("from_node_id").alias('from_n_of_citations'),F.collect_set('from_node_id').alias('from_who_cited'))
mydf3 = mydf3.withColumnRenamed('to_node_id', 'node_id_from_cit')

mydf4 = nodes.groupBy("to_node_id").agg(F.count("from_node_id").alias('to_n_of_citations'),F.collect_set('from_node_id').alias('to_who_cited'))
mydf4 = mydf4.withColumnRenamed('to_node_id', 'node_id_to_cit')


#joining everything together
other_join = newDF.join(mydf3, newDF.from_node_id == mydf3.node_id_from_cit, how='left') # Could also use 'left_outer'

other = other_join.join(mydf4, other_join.to_node_id == mydf4.node_id_to_cit, how='left') # Could also use 'left_outer'

newDF = other

newDF = newDF.na.fill(0)

#At this point we create the differnce in citations since we know which node cites who.
newDF = newDF.withColumn("diff_n_of_citations", udf_abs(struct(newDF.from_n_of_citations,
								       newDF.to_n_of_citations)))

newDF.printSchema()

#Here we are finding the number of common citation between the from_node_id and to_node_id
newDF = newDF.withColumn("n_common_citations", udf_token_overlap(struct(newDF.from_who_cited,
								       newDF.to_who_cited))) 

newDF = newDF.withColumn("link_based_JS", udf_simJaccard(struct(newDF.from_node_neighbors,
								       newDF.to_node_neighbors)))


newDF.printSchema()

#Here is the normalizer but we do not use it since we found a decrease in f1 score after applying it.

# normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
# newDF = normalizer.transform(newDF)

#in case of missing values
newDF = newDF.na.fill(0)

newDF = newDF.withColumn('features', udf_tovector(struct(
                newDF.title_overlap, newDF.authors_overlap,
                newDF.abstract_overlap, newDF.journal_overlap,
                newDF.title_simJaccard, newDF.abstract_simJaccard, newDF.journal_simJaccard,
                newDF.time_dist, newDF.same_journal,
                newDF.n_common_neigbors, newDF.link_based_JS, newDF.n_common_citations, newDF.diff_in_degree,
                 newDF.diff_n_of_citations)))


(trainDF, testDF) = newDF.randomSplit([0.7, 0.3])

# Define a Random Forest classification method.
rf = RandomForestClassifier(featuresCol='features', labelCol='label', predictionCol='pred',
                            rawPredictionCol='pred_raw',
                            maxDepth=8, maxBins=32, minInstancesPerNode=2, numTrees=50,)

# Fit the rf model using train data.
rf_model = rf.fit(trainDF)

# Transform on test data. The result is a dataframe with additional columns for the predictions.
rf_result = rf_model.transform(testDF)

# Create an evaluator to measure classification performance.
evaluator = BinaryClassificationEvaluator(rawPredictionCol='pred_raw', labelCol='label',
                                          metricName='areaUnderPR')
area_under_pr = evaluator.evaluate(rf_result)
evaluator = MulticlassClassificationEvaluator(predictionCol="pred", labelCol="label", metricName="f1")
f1_score = evaluator.evaluate(rf_result)

print("")
print("########################################################################")
print("RANDOM FOREST RESULTS")
print("Area under PR curve: " + str(area_under_pr))
print("F1 score = %g" % f1_score)
print("########################################################################")

print("Time elapsed:", time.time() - start)







