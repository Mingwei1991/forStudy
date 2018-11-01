import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.regression.FM
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import test._

object Main {
    def main(args: Array[String]): Unit = {
        println("this is a spark project")
        val lr = new LogisticRegression()
        lr.setThreshold(1)

        val pip = new Pipeline()
        val x = new FM()
        x.getThreshold
        x.getVectorSize
        val spark = SparkSession.builder().master("local").appName("test").getOrCreate()
        val a = udf {
            applist: Array[String] => applist.contains("a")
        }

        spark.udf.register("name", a)
    }
}
