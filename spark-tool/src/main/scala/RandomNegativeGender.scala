import org.apache.spark.SparkException
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{expr, split, udf}

import scala.util.Random

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @Author: Mingwei Zhang
 * @Year: 2018
 * @Version 0.1
 */

// test random choose negative sample
object RandomNegativeGender {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").appName("test").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._

    val handleNullString = udf { x: Any =>
      if (x == null) {
        "null"
      } else {
        x.toString
      }
    }

    val labelUDF = udf { label: Any =>
      if (label == null) {
        throw new SparkException("label can't be null")
      } else {
        if (label.equals("0")) 0.0 else 1.0
      }
    }

    val screenUDF1 = udf { screensize: Any =>
      if (screensize == null) {
        Double.NaN
      } else {
        val Array(x, _*) = screensize.asInstanceOf[String].split("x").map(_.toDouble)
        x
      }
    }

    val screenUDF2 = udf { screensize: Any =>
      if (screensize == null) {
        Double.NaN
      } else {
        val Array(_, x, _*) = screensize.asInstanceOf[String].split("x").map(_.toDouble)
        x
      }
    }

    val toDoubleUDF = udf { price: Any =>
      if (price == null) {
        Double.NaN
      } else {
        price.asInstanceOf[String].toDouble
      }
    }

    val publicDateUDF = udf { public_date: Any =>
      if (public_date == null) {
        Double.NaN
      } else {
        public_date.asInstanceOf[String].toDouble
      }
    }

    val gdf = udf { value: Double =>
      if (value == 1.0 && math.abs(Random.nextInt) % 10 != 1) {
        0.0
      } else {
        value
      }
    }

    val df = spark.read.format("csv")
      .option("header", "true")
      .load("/Users/mingwei/Documents/Data/Current/gender/test.csv")

    val dataset = spark.read.format("csv")
      .option("header", "true")
      .load("/Users/mingwei/Documents/Data/Current/gender/train.csv")
      .union(df)
      .union(df.filter(expr("label=1.0")))
      .select(
        labelUDF($"label").as("tag"),
        handleNullString($"brand").as("brand"),
        screenUDF1($"screensize").as("screen1"),
        screenUDF2($"screensize").as("screen2"),
        handleNullString($"model_level").as("model_level"),
        toDoubleUDF($"tot_install_apps").as("tot_install_apps"),
        split($"applist", ",").as("applist"),
        toDoubleUDF($"price").as("price"),
        publicDateUDF($"public_date").as("public_date")
      )
      .filter("size(applist)>10 and size(applist)<200").cache
    //                .sample(0.01)

    val brandIndexer = new StringIndexer().setInputCol("brand").setOutputCol("brand_index")

    val modelLevelIndexer = new StringIndexer().setInputCol("model_level").setOutputCol("model_level_index")

    val oneHot = new OneHotEncoderEstimator()
      .setInputCols(Array("brand_index", "model_level_index"))
      .setOutputCols(Array("brand_v", "model_level_v"))

    val applistVectorizer = new CountVectorizer()
      .setInputCol("applist")
      .setOutputCol("app_vector")
      .setVocabSize(3500)

    val applistVectorizer2 = new CountVectorizer()
      .setInputCol("applist")
      .setOutputCol("app_c")
      .setVocabSize(10000)

    val x = new ChiSqSelector()
      .setFeaturesCol("app_c")
      .setLabelCol("label")
      .setNumTopFeatures(500)
      .setOutputCol("apps")

    val imputer = new Imputer()
      .setStrategy("median")
      .setInputCols(Array("screen1", "screen2", "price", "tot_install_apps", "public_date"))
      .setOutputCols(Array("out_screen1", "out_screen2", "out_price", "out_tot_install_apps", "out_public_date"))

    val featureAssembler = new VectorAssembler()
      .setInputCols(Array(
        "brand_v",
        "out_screen1",
        "out_screen2",
        "model_level_v",
        "app_vector"
        //                                        "out_tot_install_apps",
        //                                        "out_price",
        //                                        "out_public_date",
        //                                        "apps"
      )
      )
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setProbabilityCol("lr_prob")
      .setPredictionCol("lr_p")
      .setMaxIter(200)
      .setElasticNetParam(0.0)
      .setRegParam(0.1)

    val featuresPipline = new Pipeline()
      .setStages(Array(
        brandIndexer,
        modelLevelIndexer,
        oneHot,
        applistVectorizer,
        imputer,
        //                                        applistVectorizer2,
        //                                        x,
        featureAssembler
      )
      )

    val featureEncoder = featuresPipline.fit(dataset)
    // feature Transform
    val Array(tmp, testDataset) = featureEncoder.transform(dataset).randomSplit(Array(0.7, 0.3))
    tmp.cache
    val t = tmp.withColumn("label", gdf($"tag"))
    val ps = t.filter("label=1.0")
    val us = t.filter("label=0.0")
    val pc = ps.count
    val uc = us.count
    val trainDataset = ps.union(us.sample(pc * 1.1 / uc))
    println(pc)
    println(uc)

    trainDataset.cache
    testDataset.cache
    val finalModel = lr.fit(trainDataset)
    val output = finalModel.transform(testDataset)
      .select($"lr_p", $"tag")
      .as[(Double, Double)].rdd

    val output_t = finalModel.transform(tmp)
      .select($"lr_p", $"tag")
      .as[(Double, Double)].rdd

    val pdf = udf { x: Vector =>
      x(1)
    }

    val testSummary = new MulticlassMetrics(output)
    val trainingSummary = new MulticlassMetrics(output_t)

    val output_p = finalModel.transform(testDataset)
      .select(pdf($"lr_prob"), $"tag")
      .as[(Double, Double)].rdd
    val test = new BinaryClassificationMetrics(output_p)
    val output_pt = finalModel.transform(trainDataset)
      .select(pdf($"lr_prob"), $"tag")
      .as[(Double, Double)].rdd
    val train = new BinaryClassificationMetrics(output_pt)




    //        val traningOfMale = trainDataset.filter("label=0").count
    //        val traningOfFemale = trainDataset.filter("label=1").count

    //        val testingOfMale = testDataset.filter("label=0").count
    //        val testingOfFemale = testDataset.filter("label=1").count

    //        println(s"model elasticNetParam: ${finalModel.getElasticNetParam}")
    //        println(s"model regParam: ${finalModel.getRegParam}")

    println(trainingSummary.confusionMatrix)
    //        println(s"amount of male in training set: $traningOfMale")
    //        println(s"amount of female in training set: $traningOfFemale")
    //        println(s"male:female = ${traningOfMale * 1.0 / traningOfFemale}")
    println(s"training accuracy: ${trainingSummary.accuracy}")
    println(s"training precision of male: ${trainingSummary.precision(0)}")
    println(s"training precision of female: ${trainingSummary.precision(1)}")
    println(s"training auc is ${train.areaUnderROC()}")
    println(s"training aupc is ${train.areaUnderPR()}")


    println(testSummary.confusionMatrix)
    //        println(s"amount of male in testing set: $testingOfMale")
    //        println(s"amount of female in testing set: $testingOfFemale")
    //        println(s"male:female = ${testingOfMale * 1.0 / testingOfFemale}")
    println(s"testing accuracy: ${testSummary.accuracy}")
    println(s"testing precision of male: ${testSummary.precision(0)}")
    println(s"testing precision of female: ${testSummary.precision(1)}")
    println(s"testing auc is ${test.areaUnderROC()}")
    println(s"testing aupc is ${test.areaUnderPR()}")


  }
}
