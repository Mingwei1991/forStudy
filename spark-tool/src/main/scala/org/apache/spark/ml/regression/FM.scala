package org.apache.spark.ml.regression

import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel

/**
  * Params for [[FM]] and [[FMModel()]].
  */
private[regression] trait FMParams extends  HasRegParam with HasElasticNetParam with HasTol
        with HasMaxIter with HasFitIntercept  with HasStandardization  with HasWeightCol
        with HasSolver with HasAggregationDepth with HasLoss with HasThreshold with Logging{

    import FM._


    /**
      * The solver algorithm for optimization.
      * Supported options: "l-bfgs" and "auto".
      * Default: "auto"
      * */
    final override val solver: Param[String] = new Param[String](
        this, "solver", "The solver algorithm for optimization. Supported options: " +
                s"${supportedSolvers.mkString(", ")}. (Default auto)",
        ParamValidators.inArray[String](supportedSolvers))

    /**
      * The loss function to be optimized.
      * Supported options: "squaredError" and "logisticLoss".
      * Default: "squaredError"
      *
      * @group param
      */
    final override val loss: Param[String] = new Param[String](
        this, "loss", "The loss function to" +
                s" be optimized. Supported options: ${supportedLosses.mkString(", ")}. (Default squaredError)",
        ParamValidators.inArray[String](supportedLosses))

    /**
      * The user defined function.
      * */
    final val UDL: Param[Double => Double] = new Param[Double => Double](
        this, "udl","User defined loss function"
    )

    final def getUDL: Double => Double = ${UDL}

    /**
      *
      */
    final val vectorSize = new IntParam(
        this, "vectorSize", "the dimension of  2-way interactions, default=8",
        ParamValidators.gt(0))

    final def getVectorSize: Int = $(vectorSize)

    /**
      * fit linear term, i.e. use 1-way interactions
      * */
    final val fitLinear = new BooleanParam(
        this, "fitLinear", "whether to fit an intercept term"
    )

    final def getFitLinear: Boolean = $(fitLinear)

    /**
      * fit bias
      * */
    final val fitBias = new BooleanParam(
        this, "fitBias", "whether to fit bias of 2-way interactions"
    )


    final def getFitBias: Boolean = $(fitBias)

    /**
      * */
    final val sigma = new DoubleParam(
        this, "sigma", s"the standard deviation of normal distribution used for initial " +
                s"parameters."
    )

    final def getSigma: Double = ${sigma}

    /**
      * Binary classification, if true the label have to be either 0.0 or 1.0
      * */
    final val forBinaryClassification = new BooleanParam(
        this, "forBinaryClassification", "whether to be a binary classification."
    )


    def getForBinaryClassification: Boolean = ${forBinaryClassification}

    /**
      * Set threshold for binary classification. valid only forBinaryClassification is true.
      * */
    def setThreshold(value: Double): this.type = {
        if (!$(forBinaryClassification)) logWarning(s"threshold is valid when forBinaryClassification is true")
        set(threshold, value)
    }

}

class FM (
                 override val uid: String
         )
        extends Regressor[Vector, FM, FMModel]
                with DefaultParamsWritable with FMParams {

    import LinearRegression._

    def this() = this(Identifiable.randomUID("fm"))

    /**
      * Set the regularization parameter.
      * Default is 0.0.
      *
      * @group setParam
      */
    def setRegParam(value: Double): this.type = set(regParam, value)
    setDefault(regParam -> 0.0)

    /**
      * Set the ElasticNet mixing parameter.
      * For alpha = 0, the penalty is an L2 penalty.
      * For alpha = 1, it is an L1 penalty.
      * For alpha in (0,1), the penalty is a combination of L1 and L2.
      * Default is 0.0 which is an L2 penalty.
      *
      * Note: Fitting under bound constrained optimization only supports L2 regularization,
      * so throws exception if this param is non-zero value.
      *
      * @group setParam
      */
    def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)
    setDefault(elasticNetParam -> 0.0)

    /**
      * Set the maximum number of iterations.
      * Default is 100.
      *
      * @group setParam
      */
    def setMaxIter(value: Int): this.type = set(maxIter, value)
    setDefault(maxIter -> 100)

    /**
      * Whether to fit an intercept term.
      * Default is true.
      *
      * @group setParam
      */
    def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)
    setDefault(fitIntercept -> true)

    def setFitLinear(value: Boolean): this.type = set(fitLinear, value)
    setDefault(fitLinear -> true)

    def setVectorSize(value: Int): this.type = set(vectorSize, value)
    setDefault(vectorSize -> 8)

    def setFitBias(value: Boolean): this.type = set(fitBias, value)
    setDefault(fitBias -> false)

    /**
      * Whether to standardize the training features before fitting the model.
      * The coefficients of models will be always returned on the original scale,
      * so it will be transparent for users. Note that with/without standardization,
      * the models should be always converged to the same solution when no regularization
      * is applied.
      * If this is not set or empty, we use standardization when the feature is sparse vector
      * while we use raw features in dense case.
      * Default is not set, so all instances have weight one.
      *
      * @group setParam
      */
    def setStandardization(value: Boolean): this.type = set(standardization, value)

    /**
      * Set the convergence tolerance of iterations.
      * Smaller value will lead to higher accuracy at the cost of more iterations.
      * Default is 1E-6.
      *
      * @group setParam
      */
    def setTol(value: Double): this.type = set(tol, value)
    setDefault(tol -> 1E-6)

    /**
      * Sets the value of param [[weightCol]].
      * If this is not set or empty, we treat all instance weights as 1.0.
      * Default is not set, so all instances have weight one.
      *
      * @group setParam
      */
    def setWeightCol(value: String): this.type = set(weightCol, value)

    def setLoss(value: String): this.type = set(loss, value)
    setDefault(loss -> SquaredError)

    /**
      *
      * */
    def setUDL(value: Double => Double): this.type = {
        if (${loss} != "udl") logWarning(s"user defined function is valid only when param loss is " +
                s"\"auto\" or \"udl\", but now loss is \"${${loss}}\"")
        set(UDL, value)
    }

    def setSolver(value: String): this.type = set(solver, value)
    setDefault(solver -> Auto)

    def setSigma(value: Double): this.type = set(sigma, value)
    setDefault(sigma -> 0.01)

    def setForBinaryClassification(value: Boolean): this.type = set(forBinaryClassification,value)
    setDefault(forBinaryClassification -> true)

    override def setThreshold(value: Double): this.type = super.setThreshold(value)
    setDefault(threshold -> 0.5)



    /**
      * Suggested depth for treeAggregate (greater than or equal to 2).
      * If the dimensions of features or the number of partitions are large,
      * this param could be adjusted to a larger size.
      * Default is 2.
      *
      * @group expertSetParam
      */
    def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)
    setDefault(aggregationDepth -> 2)


    private var optInitialModel: Option[FMModel] = None

    private def setInitialModel(model: FMModel): this.type = {
        this.optInitialModel = Some(model)
        this
    }

    protected[spark] override def train(dataset: Dataset[_]): FMModel = {
        val handlePersistence = dataset.storageLevel == StorageLevel.NONE
        train(dataset, handlePersistence)
    }

    protected[spark] def train(
                     dataset: Dataset[_],
                     handlePersistence: Boolean): FMModel = {
        val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
        val instances = dataset.select(col($(labelCol)), w, col($(featuresCol))).rdd.map {
            case Row(label: Double, weight: Double, features: Vector) =>
                Instance(label, weight, features)
        }

        if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

        val instr = Instrumentation.create(this, instances)

        instr.logParams(regParam, elasticNetParam, standardization, threshold,
            maxIter, tol, fitIntercept)

        val (summarizer, labelSummarizer) = {
            val seqOp = (c: (MultivariateOnlineSummarizer, BinaryClassSummarizer),
                         instance: Instance) =>
                (c._1.add(instance.features, instance.weight), c._2.add(instance.label, instance.weight))

            val combOp = (c1: (MultivariateOnlineSummarizer, BinaryClassSummarizer),
                          c2: (MultivariateOnlineSummarizer, BinaryClassSummarizer)) =>
                (c1._1.merge(c2._1), c1._2.merge(c2._2))

            instances.treeAggregate(
                (new MultivariateOnlineSummarizer, new BinaryClassSummarizer)
            )(seqOp, combOp, $(aggregationDepth))
        }

        val histogram = labelSummarizer.histogram // length = 2
        val numInvalid = labelSummarizer.countInvalid
        val numFeatures = summarizer.mean.size
        val numFeaturesPlusIntercept = if (getFitIntercept) numFeatures + 1 else numFeatures

        val numClasses = MetadataUtils.getNumClasses(dataset.schema($(labelCol))) match {
            case Some(n: Int) =>
                require(n == histogram.length, s"Specified number of classes $n was not 2.")
                n
            case None => histogram.length
        }

        require(histogram(0)._2 != 0.0 && histogram(1)._2 != 0.0 || !$(fitIntercept), s"All labels are the same" +
                " value and fitIntercept=true, so the coefficients will be zeros. Training is not needed.")

        if (histogram(0)._2 == 0.0) {
            instr.logInfo("The sum of the weights of negative samples is zero.")
        }

        if (histogram(1)._2 == 0.0) {
            instr.logInfo("The sum of the weights of positive samples is zero.")
        }

        instr.logInfo("Regression by FM")
        instr.logNumFeatures(numFeatures)




        new FMModel(1.toString)
    }

    override def copy(extra: ParamMap): FM = defaultCopy(extra)

}

object FM extends DefaultParamsReadable[FM] {
    // TODO
    override def load(path: String): FM = super.load(path)

    /** String name for "auto". */
    private[regression] val Auto = "auto"


    /** String name for "l-bfgs". */
    private[regression] val LBFGS = "l-bfgs"

    /** Set of solvers that LinearRegression supports. */
    private[regression] val supportedSolvers = Array(Auto, LBFGS)

    /** String name for "squaredError". */
    private[regression] val SquaredError = "squaredError"

    /** String name for "huber". */
    private[regression] val Huber = "huber"

    /** Set of loss function names that LinearRegression supports. */
    private[regression] val supportedLosses = Array(SquaredError, Huber)
}


class FMModel private[ml](
                      override val uid: String

              )
extends RegressionModel[Vector, FMModel] with FMParams {

    def setInputCol(value: String): this.type = this
    override def copy(extra: ParamMap): FMModel = ???

    override def transform(dataset: Dataset[_]): DataFrame = ???

    override def transformSchema(schema: StructType): StructType = ???

    override protected def predict(features: Vector): Double = ???
}

private[ml] class BinaryClassSummarizer extends Serializable {
    // the first element is the actual number of instances
    // the second element is weighted sum
    private var positiveClass = 0L -> 0.0
    private var negativeClass = 0L -> 0.0
    // invalid label count
    private var totalInvalidCnt: Long = 0L


    def add(label: Double, weight: Double = 1.0): this.type = {
        require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")

        if (weight == 0.0) return this

        if (label == 0.0) {
            positiveClass = (positiveClass._1 + 1, positiveClass._2 + weight)
            this
        } else if (label == 1.0) {
            negativeClass = (negativeClass._1 + 1, negativeClass._2 + weight)
            this
        } else {
            totalInvalidCnt += 1
            this
        }
    }

    def merge(that: BinaryClassSummarizer): BinaryClassSummarizer = {
        totalInvalidCnt += that.totalInvalidCnt
        positiveClass = (positiveClass._1 + that.positiveClass._1, positiveClass._2 + that.positiveClass._2)
        negativeClass = (negativeClass._1 + that.negativeClass._1, negativeClass._2 + that.negativeClass._2)
        this
    }

    /** @return The total invalid input counts. */
    def countInvalid: Long = totalInvalidCnt

    def histogram: Array[(Long, Double)] = Array(negativeClass, positiveClass)

}



