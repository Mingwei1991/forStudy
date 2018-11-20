import org.apache.spark.SparkException
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{CountVectorizer, OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.regression.FM
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Main {
    def main(args: Array[String]): Unit = {
        println("this is a spark project")

        val x = new FM()
        val spark = SparkSession.builder().master("local").appName("test").getOrCreate()
        import spark.implicits._
//        val dataf = Seq.fill[LR](1000)(LR(new DenseVector((1 to 300).map(x =>
//            if(math.random > 0.005) 0.0 else 1.0).toArray).toSparse, if(math.random > 0.4) 0.0 else 1.0)).toDF
//        x.setLabelCol("b")
//                .setFeaturesCol("a")
//                .setForBinaryClassification(false)
//                .setMaxIter(50)
//                .setFitIntercept(true)
////        val model = x.fit(dataf)
//
////        val c = model.transform(dataf)
//        val lr = new LogisticRegression()
//                .setFeaturesCol("a")
//                .setLabelCol("b")
//                .setFitIntercept(true)
//                .setPredictionCol("lrPredict")
//                .setRawPredictionCol("hi")
//
//        val lrModel = lr.fit(dataf)
//        lrModel.transform(dataf).select("hi").show
//        model.trainingSummary match {
//            case Some(summary: FMTrainingSummary) =>
//                summary.objectiveHistory.foreach(println)
//            case _ =>
//        }
        val t = Seq.fill[LR](10)(LR(new DenseVector(Array(1, 2)), 1)).toDF
        val mudf = udf { x: Any =>
            x.asInstanceOf[DenseVector].toArray(1) = 0
            1
        }

        t.select($"a", mudf(col("a")), col("b")).show
        test(spark)
    }

    def test(spark: SparkSession): Unit ={
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

        val screenUDF = udf { screensize: Any =>
            if (screensize == null) {
                Vectors.dense(Array(0.0, 0.0))
            } else {
                val Array(x, y, _*) = screensize.asInstanceOf[String].split("x").map(_.toDouble)
                Vectors.dense(Array(x, y))
            }
        }

        val toDoubleUDF = udf { price: Any =>
            if (price == null) {
                0.0
            } else {
                price.asInstanceOf[String].toDouble
            }
        }

        val publicDateUDF = udf {public_date: Any =>
            if (public_date == null) {
                0.0
            } else {
                public_date.asInstanceOf[String].toDouble
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
                .select(labelUDF($"label").as("label"), handleNullString($"brand").as("brand"),
                    screenUDF($"screensize").as("screen_v"), handleNullString($"model_level").as("model_level"), toDoubleUDF($"tot_install_apps").as("tot_install_apps"), split($"applist", ",").as("applist"),
                    toDoubleUDF($"price").as("price"), publicDateUDF($"public_date").as("public_date"))
                .filter("size(applist)>10")
                .sample(0.01)
                .cache()

        val brandIndexer = new StringIndexer().setInputCol("brand").setOutputCol("brand_index")

        val modelLevelIndexer = new StringIndexer().setInputCol("model_level").setOutputCol("model_level_index")

        val oneHot = new OneHotEncoderEstimator()
                .setInputCols(Array("brand_index", "model_level_index"))
                .setOutputCols(Array("brand_v", "model_level_v"))

        val applistVectorizer = new CountVectorizer()
                .setInputCol("applist")
                .setOutputCol("app_vector")
                .setVocabSize(3500)

        val featureAssembler = new VectorAssembler()
                .setInputCols(Array("brand_v", "screen_v", "model_level_v", "app_vector", "tot_install_apps", "price", "public_date"))
                .setOutputCol("features")

        val featuresPipline = new Pipeline()
                .setStages(Array(brandIndexer, modelLevelIndexer, oneHot, applistVectorizer, featureAssembler))



        val featureEncoder = featuresPipline.fit(dataset)
        // feature Transform
        val Array(tdf, vdf) = featureEncoder.transform(dataset).randomSplit(Array(0.8, 0.2))

        val fm = new FM().setFeaturesCol("features")
                .setForBinaryClassification(true)
                .setVectorSize(8)
                .setLabelCol("label")
                .setPredictionCol("p_label")
                .setProbabilityCol("prob")
                .setFitBias(true)
                .setFitLinear(true)
                .setFitIntercept(true)
                .setLoss("logisticLoss")
                .setElasticNetParam(0.0)
                .setRegParam(0.1)
                .setMaxIter(1)
        val fmModel = fm.fit(tdf)
        val output = fmModel.transform(vdf)
        output.printSchema()
        val traningSummary = fmModel.binarySummary
        val testSummary = fmModel.binaryEvaluate(vdf)

        println(s"training auc: ${traningSummary.areaUnderROC}")
        println(s"training accuracy: ${traningSummary.accuracy}")
        println(s"training precision of male: ${traningSummary.precision(0)}")
        println(s"training precision of female: ${traningSummary.precision(1)}")

        println(s"testing auc: ${testSummary.areaUnderROC}")
        println(s"testing accuracy: ${testSummary.accuracy}")
        println(s"testing precision of male: ${testSummary.precision(0)}")
        println(s"testing precision of female: ${testSummary.precision(1)}")
    }
}

case class LR(
                     a: Vector,
                     b: Double
             )
case class H (
             a: Int
             )


