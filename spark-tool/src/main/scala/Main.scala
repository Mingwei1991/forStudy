import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.regression.{FM, FMTrainingSummary, LinearRegression}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

object Main {
    def main(args: Array[String]): Unit = {
        println("this is a spark project")

        val x = new FM()
        val spark = SparkSession.builder().master("local").appName("test").getOrCreate()
        import spark .implicits._
        val dataf = Seq.fill[LR](1000)(LR(new DenseVector((1 to 200).map(x =>
            if(math.random > 0.05) 0.0 else 1.0).toArray).toSparse, 1.0 + math.random)).toDF
        x.setLabelCol("b")
                .setFeaturesCol("a")
                .setForBinaryClassification(false)
                .setMaxIter(200)
                .setFitIntercept(true)
        val model = x.fit(dataf)

        val c = model.transform(dataf)
        val lr = new LinearRegression()
                .setFeaturesCol("a")
                .setLabelCol("b")
                .setFitIntercept(true)
                .setPredictionCol("lrPredict")
                .setSolver("l-bfgs")
        val lrModel = lr.fit(dataf)
        lrModel.transform(c).show
        model.trainingSummary match {
            case Some(summary: FMTrainingSummary) =>
                summary.objectiveHistory.foreach(println)
            case _ =>
        }
    }
}

case class LR(
                     a: Vector,
                     b: Double
             )
case class H (
             a: Int
             )
