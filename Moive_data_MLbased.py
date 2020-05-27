# Databricks notebook source
# MAGIC %md 
# MAGIC ### Spark Moive Recommendation
# MAGIC In this notebook, we will use an Alternating Least Squares (ALS) algorithm with Spark APIs to predict the ratings for the movies in [MovieLens small dataset](https://grouplens.org/datasets/movielens/latest/)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part0: Environment setting up

# COMMAND ----------

dbutils.library.installPyPI("koalas")

# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from pyspark.sql import functions as F
%matplotlib inline
import os
import databricks.koalas as ks
os.environ["PYSPARK_PYTHON"] = "python3"

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part1: Data ETL and Data Exploration

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("moive analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# COMMAND ----------

# DBTITLE 1,Load data
movies_df = spark.read.load("/FileStore/tables/movies.csv", format='csv', header = True)
ratings_df = spark.read.load("/FileStore/tables/ratings.csv", format='csv', header = True)
links_df = spark.read.load("/FileStore/tables/links.csv", format='csv', header = True)
tags_df = spark.read.load("/FileStore/tables/tags.csv", format='csv', header = True)

# COMMAND ----------

# DBTITLE 1,Check parts of tables
movies_df.show(5)
ratings_df.show(5)
links_df.show(5)
tags_df.show(5)

# COMMAND ----------

# DBTITLE 1,Users' Information
# get statistics for users
users_result = ratings_df.groupBy("userId").count()\
.orderBy('count', ascending=False)

fig = plt.figure()
tmp=sns.distplot(users_result.toPandas()['count'], hist=True)
plt.ylabel('percentage')
plt.xlabel('number of ratings')
tmp.set_title('Distribution of number of ratings (per user)')
display(tmp)

# COMMAND ----------

# DBTITLE 1,Movies' Information
# get statistics for movies
movies_result=ratings_df.groupBy("movieId").count()\
.orderBy('count', ascending=False)

fig = plt.figure()
tmp=sns.distplot(movies_result.toPandas()['count'], color = 'burlywood', hist=True)
plt.ylabel('percentage')
plt.xlabel('number of ratings')
tmp.set_title('Distribution of number of ratings (per movie)')
display(tmp)

# COMMAND ----------

# analysis movies by content
movies_genres_temp=movies_df.where('genres is not null')\
.select('title', F.explode(F.split('genres', '\|'))\
.alias('genres'))

movies_genres=movies_genres_temp.groupBy('genres').count()\
.orderBy('count', ascending=False)\
.toPandas()

plt.figure(figsize=(60,10))
tmp=sns.barplot(x='genres', y='count', data=movies_genres)
tmp.set_title('Number of movies of each category')
display(tmp)

# COMMAND ----------

# DBTITLE 1,Ratings' Information
## get statistics for ratings
ratings_result=ratings_df.groupBy('rating').count()\
.withColumn('rating', ratings_df.rating.cast('float'))\
.toPandas()

fig = plt.figure()
tmp=sns.barplot(x='rating', y='count', data=ratings_result)
tmp.set_title('Number of movies of each rating')
display(tmp)

# COMMAND ----------

# DBTITLE 1,Unrated movies
rated_list=ratings_df.select(ratings_df.movieId)\
.where('rating is not null')\
.withColumnRenamed('movieId', 'rated').distinct()

unrated_list=movies_df.join(rated_list, movies_df.movieId==rated_list.rated, "left_outer")\
.where('rated is null')

print('{} movies are unrated, {} movies are rated'.format(unrated_list.count(), rated_list.count()))

unrated_list.toPandas().fillna('No')

# COMMAND ----------

# DBTITLE 1,Max, Min ratings per user
tmp1 = ratings_df.groupBy("userID").count().toPandas()['count'].min()
tmp2 = ratings_df.groupBy("movieId").count().toPandas()['count'].min()
print('For the users that rated movies and the movies that were rated:')
print('Minimum number of ratings per user is {}'.format(tmp1))
print('Minimum number of ratings per movie is {}'.format(tmp2))

# COMMAND ----------

# DBTITLE 1,Percentage of movies with only one rate
tmp1 = sum(ratings_df.groupBy("movieId").count().toPandas()['count'] == 1)
tmp2 = ratings_df.select('movieId').distinct().count()
print('{} out of {} movies are rated by only one user'.format(tmp1, tmp2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark SQL and OLAP 

# COMMAND ----------

movies_df.registerTempTable("movies")
ratings_df.registerTempTable("ratings")
links_df.registerTempTable("links")
tags_df.registerTempTable("tags")

# COMMAND ----------

# MAGIC %md ### Q1: The number of Users

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(distinct userID) as Number_of_user
# MAGIC from ratings

# COMMAND ----------

# MAGIC %md ### Q2: The number of Movies

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(distinct movieID) as Number_of_movie
# MAGIC from movies

# COMMAND ----------

# MAGIC %md ### Q3:  How many movies are rated by users? List movies not rated before

# COMMAND ----------

# MAGIC %sql
# MAGIC select title, genres
# MAGIC from movies
# MAGIC where movieId not in (
# MAGIC   select movieId
# MAGIC   from ratings)

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(distinct movieId) as Number_movies_are_rated_by_users
# MAGIC from ratings

# COMMAND ----------

# MAGIC %md ### Q4: List Movie Genres

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct category
# MAGIC from movies
# MAGIC lateral view explode(split(genres, '[|]')) as category
# MAGIC order by category

# COMMAND ----------

# MAGIC %md ### Q5: Movie for Each Category

# COMMAND ----------

# MAGIC %sql
# MAGIC select category as Category, count(movieId) as number
# MAGIC from movies lateral view explode(split(genres, '[|]')) as category
# MAGIC group by category
# MAGIC order by number desc

# COMMAND ----------

# MAGIC %sql
# MAGIC select t.Category, concat_ws(',', collect_set(t.title)) as list_of_movies
# MAGIC from (
# MAGIC   select Category, title
# MAGIC   from movies
# MAGIC   lateral view explode(split(genres, '[|]')) as Category) as t
# MAGIC group by t.Category

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part2: Spark ALS based approach for training model
# MAGIC We will use an Spark ML to predict the ratings, here reload "ratings.csv" using ``sc.textFile`` and then convert it to the form of (user, item, rating) tuples.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preprocessing

# COMMAND ----------

ratings_df.show()

# COMMAND ----------

movie_ratings=ratings_df.drop('timestamp')

# COMMAND ----------

# Data type convert
from pyspark.sql.types import IntegerType, FloatType
movie_ratings = movie_ratings.withColumn("userId", movie_ratings["userId"].cast(IntegerType()))
movie_ratings = movie_ratings.withColumn("movieId", movie_ratings["movieId"].cast(IntegerType()))
movie_ratings = movie_ratings.withColumn("rating", movie_ratings["rating"].cast(FloatType()))

# COMMAND ----------

movie_ratings.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### ALS Model Selection and Evaluation
# MAGIC 
# MAGIC With the ALS model, we can use a grid search to find the optimal hyperparameters.

# COMMAND ----------

# import package
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder

# COMMAND ----------

#Create test and train set
(training,test)=movie_ratings.randomSplit([0.8,0.2], seed = 2020)

# COMMAND ----------

#Create ALS model
als = ALS(maxIter=5, rank=10, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")

# COMMAND ----------

#Tune model using ParamGridBuilder
paramGrid = ParamGridBuilder()\
            .addGrid(als.regParam, [0.1, 0.01, 0.001])\
            .addGrid(als.maxIter, [3, 5, 10])\
            .addGrid(als.rank, [5, 10, 15])\
            .build()

# COMMAND ----------

# Define evaluator as RMSE
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

# COMMAND ----------

# Build Cross validation 
crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# COMMAND ----------

#Fit ALS model to training data
cvModel = crossval.fit(training)
predictions = cvModel.transform(training)
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# COMMAND ----------

#Extract best model from the tuning exercise using ParamGridBuilder
best_model = cvModel.bestModel
predictions=best_model.transform(test)
rmse = evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Model testing
# MAGIC And finally, make a prediction and check the testing error.

# COMMAND ----------

#Generate predictions and evaluate using RMSE
predictions=best_model.transform(test)
rmse = evaluator.evaluate(predictions)

# COMMAND ----------

#Print evaluation metrics and model parameters
print ("RMSE = "+str(rmse))
print ("**Best Model**")
print (" Rank:"+str(best_model._java_obj.parent().getRank())), 
print (" MaxIter:"+str(best_model._java_obj.parent().getMaxIter())), 
print (" RegParam:"+str(best_model._java_obj.parent().getRegParam())), 

# COMMAND ----------

predictions.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part3: Model apply and see the performance

# COMMAND ----------

# DBTITLE 1,All Data
alldata=best_model.transform(movie_ratings)
rmse = evaluator.evaluate(alldata)
print ("RMSE = "+str(rmse))

# COMMAND ----------

alldata.registerTempTable("alldata")

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from alldata

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * 
# MAGIC from movies join alldata on movies.movieId=alldata.movieId

# COMMAND ----------

# DBTITLE 1,For the movie with most number of ratings
#plot actual and predicted ratings for movie with most number of ratings
movie_rated_most=alldata.where('movieId = 356').select('rating', 'prediction').orderBy('rating')

RMSE_evaluator_rounded = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
print(RMSE_evaluator_rounded.evaluate(movie_rated_most))

# COMMAND ----------

movie_rated_most=movie_rated_most.toPandas()
abs(movie_rated_most.prediction-movie_rated_most.rating).mean()

# COMMAND ----------

fig = plt.figure()
plt.plot(movie_rated_most.index, movie_rated_most.rating, 'x', label='rating')
plt.plot(movie_rated_most.index, movie_rated_most.prediction, '.', label='predictions')
plt.xlabel('index')
plt.ylabel('ratings')
plt.legend()
plt.title('Actual & predicted ratings for movie with 329 ratings')
display(fig)

# COMMAND ----------

# DBTITLE 1,movies with only one rating
#plot actual and predicted ratings for movie with 1 ratings
movie_rated_least=alldata.join(movies_result, movies_result.movieId==alldata.movieId, 'left')\
.where('count=1').select('rating', 'prediction').orderBy('rating')

print(RMSE_evaluator_rounded.evaluate(movie_rated_least))

# COMMAND ----------

movie_rated_least=movie_rated_least.toPandas()

fig = plt.figure()
plt.plot(movie_rated_least.index, movie_rated_least.rating, '*', label='rating')
plt.plot(movie_rated_least.index, movie_rated_least.prediction, '-', label='predictions')
plt.xlabel('index')
plt.ylabel('ratings')
plt.legend()
plt.title('Actual & predicted ratings for movies with 1 ratings')
display(fig)

# COMMAND ----------

# DBTITLE 1,User rated most movies
#plot actual and predicted ratings for most active user
user_rate_most=alldata.where('userId = 414').select('rating', 'prediction').orderBy('rating')

print(RMSE_evaluator_rounded.evaluate(user_rate_most))

# COMMAND ----------

user_rate_most=user_rate_most.toPandas()

fig = plt.figure()
plt.plot(user_rate_most.index, user_rate_most.rating, 'X', label='rating')
plt.plot(user_rate_most.index, user_rate_most.prediction, '.', label='predictions')
plt.xlabel('index')
plt.ylabel('ratings')
plt.legend()
plt.title('Actual & predicted ratings for user rated 2698 movies')
display(fig)

# COMMAND ----------

# DBTITLE 1,Users only rated 20 movies
#plot actual and predicted ratings for least active user
user_pool=users_result.where('count=20').toPandas()['userId'].tolist()
user_rate_least=alldata.where(alldata.movieId.isin(user_pool)).select('rating', 'prediction').orderBy('rating')

print(RMSE_evaluator_rounded.evaluate(user_rate_least))

# COMMAND ----------

user_rate_least=user_rate_least.toPandas()


fig = plt.figure()
plt.plot(user_rate_least.index, user_rate_least.rating, 'x', label='rating')
plt.plot(user_rate_least.index, user_rate_least.prediction, '.', label='predictions')
plt.xlabel('index')
plt.ylabel('ratings')
plt.legend()
plt.title('Actual & predicted ratings for users rated 20 movies')
display(fig)

# COMMAND ----------

# DBTITLE 1,Find general threshold for numbers of ratings to stable prediction
#plot average abs error by each movie and number of ratings
average_by_movie=alldata.groupBy('movieId')\
.agg(F.mean(F.abs(alldata.rating-alldata.prediction)).alias('MEBM'), F.count('rating').alias('number_of_users_rated'))\
.orderBy('MEBM', ascending=False)

# COMMAND ----------

tmp=average_by_movie.toPandas()
fig = plt.figure()
tmp=sns.scatterplot(x='number_of_users_rated', y='MEBM', data=tmp)
tmp.set_title('average abs error (per movies)')
display(tmp)

# COMMAND ----------

#plot average abs error by number of ratings
average_by_rating_number=alldata.join(movies_result, movies_result.movieId==alldata.movieId, 'left')\
.groupBy('count')\
.agg(F.mean(F.abs(alldata.rating-alldata.prediction)).alias('MEBC'))\
.orderBy('count', ascending=False)

# COMMAND ----------

tmp=average_by_rating_number.toPandas()
fig = plt.figure()
tmp=sns.scatterplot(x='count', y='MEBC', data=tmp)
tmp.set_title('average abs error by number of ratings')
display(tmp)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part4: Applications
# MAGIC ### Recommend moive to users with id: 575, 232. 
# MAGIC you can choose some users to recommend the moives 

# COMMAND ----------

userRecommend = best_model.recommendForAllUsers(10)

# COMMAND ----------

display(userRecommend)

# COMMAND ----------

user_recommendation = userRecommend.to_koalas()
user_recommendation.head()

# COMMAND ----------

movies_koalas = movies_df.to_koalas()

# COMMAND ----------

def movie_recommendation(user_recommendation, userId, movies_koalas):
  rec_movieId = []
  for item in user_recommendation.loc['userId' == userId][1]:
    rec_movieId.append(item[0])
  return movies_koalas.loc[movies_koalas.movieId.isin(rec_movieId)]

# COMMAND ----------

movie_recommendation(user_recommendation, 575, movies_koalas)

# COMMAND ----------

movie_recommendation(user_recommendation, 232, movies_koalas)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find the similar moives for moive with id: 463, 471
# MAGIC You can find the similar moives based on the ALS results

# COMMAND ----------

item_factors = best_model.itemFactors
movie_factors = item_factors.to_koalas()
movie_factors.head()

# COMMAND ----------

def similar_movies(features, movieId):

  try: 
    target_id_feature = movie_factors.loc[movie_factors.id == movieId].features.to_numpy()[0]
  except:
    return 'There is no movie with id ' + str(movieId)

  similarities = []
  for feature in movie_factors['features'].to_numpy():
    similarity = np.dot(target_id_feature,feature)/(np.linalg.norm(target_id_feature) * np.linalg.norm(feature))
    similarities.append(similarity)
    
  ks_similarity = ks.DataFrame({'similarity' : similarities}, index = movie_factors.id.to_numpy())
  # top 11 similar movies contain the movie itself with similarity = 1, so I need to remove it. 
  top_11 = ks_similarity.sort_values(by = ['similarity'], ascending = False).head(11)
  joint = top_11.merge(movies_koalas, left_index=True, right_on = 'movieId', how = 'inner')
  joint.sort_values(by = ['similarity'], ascending = False,inplace = True)
  joint.reset_index(inplace = True)
  # take top 10 similar movies
  return joint.loc[1:,['movieId','title','genres','similarity']]

# COMMAND ----------

similar_movies(features = movie_factors['features'], movieId = 463)

# COMMAND ----------

similar_movies(features = movie_factors['features'], movieId = 471)
