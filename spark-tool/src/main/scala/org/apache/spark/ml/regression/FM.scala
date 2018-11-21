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

package org.apache.spark.ml.regression

import java.util.Locale

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{LBFGS => BreezeLBFGS, OWLQN => BreezeOWLQN}
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{DenseVector, Matrices, Matrix, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.optim.aggregator.{LeastSquaresFMAggregator, LogisticFMAggregator}
import org.apache.spark.ml.optim.loss.{L2Regularization, RDDLossFunction}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.stat.{MultivariateOnlineSummarizer, MultivariateStatisticalSummary}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataType, DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.VersionUtils

import scala.collection.mutable
import scala.util.Random

/**
  * Params for [[FM]] and [[FMModel()]].
  * Factorization Machine
  * See: Steffen Rendle, Factorization Machines, in Proceedings of the 10th IEEE International Conference
  * on Data Mining (ICDM 2010), Sydney, Australia.
  */
private[regression] trait FMParams extends HasRegParam with HasElasticNetParam with HasTol
        with HasMaxIter with HasFitIntercept with HasWeightCol // with HasSolver
        with HasAggregationDepth with HasLoss with HasThreshold with Logging {

    import FM._


    /**
      * The solver algorithm for optimization.
      * Supported options: "l-bfgs" and "auto".
      * Default: "auto"
      **/
    /*final override val solver: Param[String] = new Param[String](
        this, "solver", "The solver algorithm for optimization. Supported options: " +
                s"${supportedSolvers.mkString(", ")}. (Default auto)",
        ParamValidators.inArray[String](supportedSolvers))*/

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
      *
      */
    final val vectorSize = new IntParam(
        this, "vectorSize", "the dimension of  2-way interactions, default=8",
        ParamValidators.gt(0))
    /**
      * fit bias
      **/
    final val fitBias = new BooleanParam(
        this, "fitBias", "whether to fit bias of 2-way interactions"
    )
    /**
      * fit linear term, i.e. use 1-way interactions
      **/
    final val fitLinear = new BooleanParam(
        this, "fitLinear", "whether to fit an intercept term"
    )
    /**
      * stdev for initialization of 2-way factors; default=0.1
      **/
    final val sigma = new DoubleParam(
        this, "sigma", s"the standard deviation for initialization of 2-way factors ")
    /**
      * Binary classification, if true the label have to be either 0.0 or 1.0
      **/
    final val forBinaryClassification = new BooleanParam(
        this, "forBinaryClassification", "whether to be a binary classification."
    )

    /**
      * Param for Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities.
      *
      * @group param
      */
    final val probabilityCol: Param[String] = new Param[String](this, "probabilityCol", "Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities")

    final val bin: IntParam = new IntParam(this, "bin",
        s"""
           |if greater than 0, then the curves (ROC curve, PR curve) computed internally will be down-sampled to this many "bins". If 0, no down-sampling will occur.
         """.stripMargin)


    final def getVectorSize: Int = $(vectorSize)

    final def getFitBias: Boolean = $(fitBias)

    final def getFitLinear: Boolean = $(fitLinear)

    def getSigma: Double = $(sigma)

    def getForBinaryClassification: Boolean = $(forBinaryClassification)

    def getProbabilityCol: String = $(probabilityCol)

    def getBin: Int = $(bin)

}

// TODO load a pre-trained model for retrain
// TODO implement setStandardization
class FM(
                override val uid: String
        )
        extends Regressor[Vector, FM, FMModel]
                with DefaultParamsWritable with FMParams {

    import FM._

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

    def setVectorSize(value: Int): this.type = set(vectorSize, value)

    setDefault(vectorSize -> 8)

    def setFitBias(value: Boolean): this.type = set(fitBias, value)

    setDefault(fitBias -> false)

    def setFitLinear(value: Boolean): this.type = set(fitLinear, value)

    setDefault(fitLinear -> true)

    /**
      * Whether to fit an intercept term.
      * Default is true.
      *
      * @group setParam
      */
    def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)

    setDefault(fitIntercept -> true)

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

    def setSigma(value: Double): this.type = {
        if (!$(forBinaryClassification)) logWarning(s"Sigma is only valid when forBinaryClassification is " +
                s"true, but now is false.")
        set(sigma, value)
    }
    setDefault(sigma -> 0.01)

    override def getSigma: Double = {
        if (!$(forBinaryClassification)) logWarning(s"Sigma is only valid when forBinaryClassification is " +
                s"true, but now is false.")
        $(sigma)
    }

    def setForBinaryClassification(value: Boolean): this.type = set(forBinaryClassification, value)

    setDefault(forBinaryClassification -> false)

    /**
      * Set threshold for binary classification. valid only forBinaryClassification is true.
      **/
    def setThreshold(value: Double): this.type = {
        if (!$(forBinaryClassification)) logWarning(s"Threshold is only valid when forBinaryClassification " +
                s"is true, but now is false.")
        set(threshold, value)
    }

    setDefault(threshold -> 0.5)

    override def getThreshold: Double = {
        if (!$(forBinaryClassification)) logWarning(s"Threshold is only valid when forBinaryClassification " +
                s"is true, but now is false.")
        $(threshold)
    }

    def setProbabilityCol(value: String): this.type = {
        if (!$(forBinaryClassification)) logWarning(s"Probability column is only valid when " +
                s"forBinaryClassification is true, but now is false.")
        set(probabilityCol, value)
    }
    setDefault(probabilityCol, "probability")

    override def getProbabilityCol: String = {
        if (!$(forBinaryClassification)) logWarning(s"Probability column is only valid when " +
                s"forBinaryClassification is true, but now is false.")
        $(probabilityCol)
    }

    def setBin(value: Int): this.type = {
        if (!$(forBinaryClassification)) logWarning(s"Bin is only valid when " +
                s"forBinaryClassification is true, but now is false.")
        set(bin, value)
    }

    override def getBin: Int = {
        if (!$(forBinaryClassification)) logWarning(s"Bin is only valid when forBinaryClassification is " +
                s"true, but now is false.")
        $(bin)
    }


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

    override def copy(extra: ParamMap): FM = defaultCopy(extra)

    override def save(path: String): Unit = super.save(path)

    // TODO check parameters
    override def validateAndTransformSchema(schema: StructType,
                                            fitting: Boolean, featuresDataType: DataType): StructType = {
        val parentSchema = super.validateAndTransformSchema(schema, fitting, featuresDataType)
        if ($(forBinaryClassification) && $(probabilityCol).isEmpty) {
           SchemaUtils.appendColumn(parentSchema, $(probabilityCol), new VectorUDT)
        } else {
            parentSchema
        }
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

        instr.logParams(vectorSize, fitIntercept, fitLinear, fitBias, regParam, elasticNetParam, weightCol,
            forBinaryClassification, threshold, loss, sigma, aggregationDepth, maxIter, tol) //solver

        val featureSummarizer = if ($(forBinaryClassification)) {
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
            val numInvalid = labelSummarizer.countInvalid
            // require numInvalid=0
            if (numInvalid != 0) {
                val msg = s"Classification labels should be in 0.0 or 1.0. " +
                        s"Found $numInvalid invalid labels."
                logError(msg)
                throw new SparkException(msg)
            }

            val histogram = labelSummarizer.histogram // length = 2
            val numClasses = MetadataUtils.getNumClasses(dataset.schema($(labelCol))) match {
                case Some(n: Int) =>
                    require(n == 1 || n == 2, s"Binary classification  support 1 or 2 outcome class " +
                            s"but find $n.")
                    n
                case None => 2
            }

            instr.logNumClasses(numClasses)
            // if fit intercept, labels can not be constant
            require(histogram(0)._2 != 0.0 && histogram(1)._2 != 0.0 || !$(fitIntercept), s"All labels are the same" +
                    " value and fitIntercept=true, so the coefficients will be zeros. Training is not needed.")

            if (histogram(0)._2 == 0.0) logWarning("The sum of the weights of negative samples is zero.")
            if (histogram(1)._2 == 0.0) logWarning("The sum of the weights of positive samples is zero.")
            summarizer
        } else {
            val summarizer = {
                val seqOp = (c: MultivariateOnlineSummarizer, instance: Instance) =>
                    c.add(instance.features, instance.weight)

                val combOp = (c1: MultivariateOnlineSummarizer, c2: MultivariateOnlineSummarizer) =>
                    c1.merge(c2)

                instances.treeAggregate(new MultivariateOnlineSummarizer)(seqOp, combOp, $(aggregationDepth))
            }
            summarizer
        }

        instr.logNamedValue("sample size", featureSummarizer.count)

        val numFeatures = instances.first().features.size
        instr.logNumFeatures(numFeatures)

        val numNonConstantFeatures = featureSummarizer.variance.toArray.count(_ == 0.0)
        if (numNonConstantFeatures != 0)
            logWarning(s"There are $numNonConstantFeatures features with constant values.")

        val numCoefficients = {
            var t = $(vectorSize) * numFeatures
            if ($(fitBias)) t += numFeatures
            if ($(fitLinear)) t += numFeatures
            if ($(fitIntercept)) t += 1
            t
        }
        instr.logNamedValue("coefficients size", numCoefficients)

        // regularization parameter
        val regParamL1 = $(elasticNetParam) * $(regParam)
        val regParamL2 = (1.0 - $(elasticNetParam)) * $(regParam)

        // TODO support custom regularization
        // add L2 regularization
        val l2Regularization = if (regParamL2 != 0.0) {
            val u = if (getFitIntercept) numCoefficients else numCoefficients - 1
            val shouldApply: Int => Boolean = idx => idx >= 0 && idx < u
            Some(new L2Regularization(regParamL2, shouldApply, None))
        } else {
            None
        }

        // cost function aka loss function
        // TODO add more loss
        val costFun = getLoss.toLowerCase(Locale.ROOT) match {
            case "logisticloss" =>
                instr.logNamedValue("Loss function", LogisticLoss)
                val getAggregatorFunc = new LogisticFMAggregator(numFeatures, $(vectorSize), $(fitBias), $(fitLinear), $(fitIntercept))(_)
                new RDDLossFunction(instances, getAggregatorFunc, l2Regularization, $(aggregationDepth))
            case "squarederror" =>
                instr.logNamedValue("loss function", SquaredError)
                val getAggregatorFunc = new LeastSquaresFMAggregator(numFeatures, $(vectorSize), $(fitBias), $(fitLinear), $(fitIntercept))(_)
                new RDDLossFunction(instances, getAggregatorFunc, l2Regularization, $(aggregationDepth))
            case _ =>
                throw new IllegalArgumentException(s"$getLoss is not supported now.")
        }

        // select optimize according to L1 regularization
        // The memory used in L-BFGS or OWLQN
        val memory = 10
        // TODO support more optimizers
        val optimizer = if (regParamL1 == 0.0) {
            new BreezeLBFGS[BDV[Double]]($(maxIter), memory, $(tol))
        } else {
            val position = if ($(fitBias)) numFeatures * ($(vectorSize) + 1) else numFeatures * $(vectorSize)

            def regParamL1Fun: Int => Double = index => {
                // Remove the L1 penalization on the intercept and latent coefficients
                val isLinearTerm = $(fitLinear) && position <= index && index < position + numFeatures
                if (isLinearTerm) {
                    regParamL1
                } else {
                    0.0
                }
            }

            new BreezeOWLQN[Int, BDV[Double]]($(maxIter), memory, regParamL1Fun, $(tol))
        }

        // initial coefficients
        val coefficientVector = {
            val rand = new Random()
            val t = if ($(fitBias)) (1 + $(vectorSize)) * numFeatures else $(vectorSize) * numFeatures
            new DenseVector(Array.fill(t)(rand.nextGaussian() * $(sigma))
                    ++ Array.fill(numCoefficients - t)(0.0))
        }

        val states = optimizer.iterations(costFun, coefficientVector.asBreeze.toDenseVector)


        val arrayBuilder = mutable.ArrayBuilder.make[Double]
        var state: optimizer.State = null
        var i = 0
        while (states.hasNext) {
            state = states.next()
            arrayBuilder += state.adjustedValue
            i += 1
        }
        if (state == null) {
            val msg = s"${optimizer.getClass.getName} failed. The optimizer iterated $i times."
            logError(msg)
            throw new SparkException(msg)
        }

        val parameters = state.x.toArray.clone()
        var t = numFeatures * $(vectorSize)
        val latentVectors = parameters.slice(0, t)
        val biases: Array[Double] = if ($(fitBias)) {
            t += numFeatures
            parameters.slice(t - numFeatures, t)
        } else {
            null
        }
        val linearCoefficients: Array[Double] = if ($(fitLinear)) {
            parameters.slice(t, t + numFeatures)
        } else {
            null
        }
        val interceptCoefficient = if ($(fitIntercept)) {
            parameters(numCoefficients - 1)
        } else {
            0.0
        }

        if (handlePersistence) instances.unpersist()

        // parents are not set in class LogisticRegression and LinearRegression.
        val model = copyValues(new FMModel(uid, numFeatures, latentVectors, biases,
            linearCoefficients, interceptCoefficient)).setParent(this)

        val trainingSummary: FMTrainingSummary = if ($(forBinaryClassification)) {
            // Handle possible missing or invalid probability columns and prediction columns
            val (summaryModel, probabilityColName, predictionColName) = model.findBinarySummaryModel
            new BinaryFMTrainingSummaryImpl(
                summaryModel.transform(dataset),
                probabilityColName,
                predictionColName,
                $(labelCol),
                $(featuresCol),
                10,
                arrayBuilder.result
            )
        } else {
            // Handle possible missing or invalid prediction columns
            val (summaryModel, predictionColName) = model.findRegressionSummaryModel

            new RegressionFMTrainingSummaryImpl(
                summaryModel.transform(dataset),
                predictionColName,
                $(labelCol),
                $(featuresCol),
                summaryModel,
                arrayBuilder.result
            )
        }

        model.setSummary(Some(trainingSummary))

        instr.logSuccess(model)
        model
    }

}

object FM extends DefaultParamsReadable[FM] {
    /** String name for "auto". */
    private[regression] val Auto = "auto"
    /** String name for "l-bfgs". */
    private[regression] val LBFGS = "l-bfgs"
    /** Set of solvers that LinearRegression supports. */
    private[regression] val supportedSolvers = Array(Auto, LBFGS)
    /** String name for "squaredError". */
    private[regression] val SquaredError = "squaredError"
    /** String name for "logisticLoss". */
    private[regression] val LogisticLoss = "logisticLoss"
    /** Set of loss function names that LinearRegression supports. */
    private[regression] val supportedLosses = Array(SquaredError, LogisticLoss)

    // TODO
    override def load(path: String): FM = super.load(path)

    override def read: MLReader[FM] = super.read
}


class FMModel private[ml](
                                 override val uid: String,
                                 override val numFeatures: Int,
                                 val latentVectors: Array[Double],
                                 val biases: Array[Double],
                                 val linearCoefficients: Array[Double],
                                 val interceptCoefficient: Double
                         )
        extends RegressionModel[Vector, FMModel] with FMParams with MLWritable {
    // TODO

    def setProbabilityCol(value: String): this.type = {
        if (!$(forBinaryClassification)) {
            throw new SparkException(s"Probability column is only valid for FM binary classification Model, " +
                    s"but this is an FM regression model.")
        }
        set(probabilityCol, value)
    }

    override def getProbabilityCol: String = {
        if (!$(forBinaryClassification)) {
            throw new SparkException(s"Probability column is only valid for FM binary classification Model, " +
                    s"but this is an FM regression model.")
        }
        $(probabilityCol)
    }

    def setThreshold(value: Double): this.type = {
        if (!$(forBinaryClassification)) {
            throw new SparkException(s"Threshold is only valid for FM binary classification Model, " +
                    s"but this is an FM regression model.")
        }
        set(threshold, value)
    }

    override def getThreshold: Double= {
        if (!$(forBinaryClassification)) {
            throw new SparkException(s"Threshold is only valid for FM binary classification Model, " +
                    s"but this is an FM regression model.")
        }
        $(threshold)
    }

    def setBin(value: Int): this.type = {
        if (!$(forBinaryClassification)) {
            throw new SparkException(s"Bin is only valid for FM binary classification Model, " +
                    s"but this is an FM regression model.")
        }
        set(bin, value)
    }
    setDefault(bin, 0)
    override def getBin: Int = {
        if (!$(forBinaryClassification)) {
            throw new SparkException(s"Bin is only valid for FM binary classification Model, " +
                    s"but this is an FM regression model.")
        }
        $(bin)
    }

    override def getSigma: Double = {
        if (!$(forBinaryClassification)) {
            throw new SparkException(s"Sigma is only valid for FM binary classification Model, " +
                    s"but this is an FM regression model.")
        }
        $(sigma)
    }

    @transient var trainingSummary: Option[FMTrainingSummary] = None

    private[regression] def setSummary(summary: Option[FMTrainingSummary]): this.type = {
        this.trainingSummary = summary
        this
    }

    /** Indicates whether a training summary exists for this model instance. */
    def hasSummary: Boolean = trainingSummary.isDefined

    def binarySummary: BinaryFMTrainingSummary = {
        if (!$(forBinaryClassification) || trainingSummary.isEmpty) {
            throw new SparkException(s"There is no training binary summary.")
        }
        trainingSummary match {
            case Some(x: BinaryFMTrainingSummary) => x
            case _ => throw new SparkException("There is no training binary summary.")
        }
    }

    def regressionSummary: RegressionFMTrainingSummary = {
        if ($(forBinaryClassification) || trainingSummary.isEmpty) {
            throw new SparkException(s"There is no training regression summary.")
        }
        trainingSummary match {
            case Some(x: RegressionFMTrainingSummary) => x
            case _ => throw new SparkException("There is no training regression summary.")
        }
    }

    def binaryEvaluate(dataset: Dataset[_]): BinaryFMSummary = {
        // Handle possible missing or invalid probability columns and prediction columns
        val (summaryModel, probabilityColName, predictionColName) = findBinarySummaryModel
        new BinaryFMSummaryImpl(
            summaryModel.transform(dataset),
            probabilityColName,
            predictionColName,
            $(labelCol),
            $(featuresCol),
            getBin
        )
    }

    def regressionEvaluate(dataset: Dataset[_]): RegressionFMSummary = {
        // Handle possible missing or invalid prediction columns
        val (summaryModel, predictionColName) = findRegressionSummaryModel

        new RegressionFMSummaryImpl(
            summaryModel.transform(dataset),
            predictionColName,
            $(labelCol),
            $(featuresCol),
            summaryModel
        )
    }

    override def validateAndTransformSchema(schema: StructType,
                                            fitting: Boolean, featuresDataType: DataType): StructType = {
        val parentSchema = super.validateAndTransformSchema(schema, fitting, featuresDataType)
        if ($(forBinaryClassification)) {
            SchemaUtils.appendColumn(parentSchema, $(probabilityCol), new VectorUDT)
        } else {
            parentSchema
        }
    }

    override def transform(dataset: Dataset[_]): DataFrame = {
        transformSchema(dataset.schema, logging = true)
        var outputData = dataset
        var numColsOutput = 0
        if ($(forBinaryClassification)) {
            if ($(probabilityCol).nonEmpty) {
                val probCol = udf(probability _).apply(col(getFeaturesCol))
                outputData = outputData.withColumn($(probabilityCol), probCol)
                numColsOutput += 1
            }
            if ($(predictionCol).nonEmpty) {
                val predCol = if ($(probabilityCol).nonEmpty) {
                    udf(probability2prediction _ ).apply(col($(probabilityCol)))
                } else {
                    udf(raw2prediction _ ).apply(col(getFeaturesCol))
                }
                outputData = outputData.withColumn($(predictionCol), predCol)
                numColsOutput += 1
            }
        } else {
            if ($(predictionCol).nonEmpty) {
                val predCol = udf(predictionRaw _).apply(col(getFeaturesCol))

                outputData = outputData.withColumn($(predictionCol), predCol)
                numColsOutput += 1
            }
        }
        if (numColsOutput == 0) {
            this.logWarning(s"$uid: FMModel.transform() was called as NOOP" +
                    " since no output columns were set.")
        }
        outputData.toDF
    }

//    override def transformSchema(schema: StructType): StructType = ???
    protected def predictionRaw(features: Vector): Double = {
        val size = $(vectorSize)
        var originPosition = 0
        var i = 0
        val vectorSum = new Array[Double](size)
        var combinationSum = 0.0
        features.foreachActive { (index, value) =>
            if (value != 0.0) {
                originPosition = index * size
                var norm2 = 0.0
                while (i < size) {
                    val coefficientPosition = originPosition + i
                    vectorSum(i) += value * latentVectors(coefficientPosition)
                    norm2 += latentVectors(coefficientPosition) * latentVectors(coefficientPosition)
                    i += 1
                }
                combinationSum -= value * value * norm2
                i = 0
            }
        }

        while (i < size) {
            combinationSum += vectorSum(i) * vectorSum(i)
            i += 1
        }
        i = 0
        combinationSum /= 2.0
        originPosition = numFeatures * size

        if ($(fitBias) && biases != null) {
            var sum = 0.0
            features.foreachActive { (index, value) =>
                if (value != 0.0) {
                    val t = biases(index) * value
                    sum += t
                    combinationSum -= t * t / 2.0
                }

            }
            combinationSum += sum * sum / 2
        }

        if ($(fitLinear) && linearCoefficients != null) {
            features.foreachActive { (index, value) =>
                if (value != 0.0) {
                    combinationSum += linearCoefficients(index) * value
                }
            }
        }

        if ($(fitIntercept)) {
            combinationSum += interceptCoefficient
        }
        combinationSum
    }

    protected def probability(features: Vector): Vector = {
        val m = predictionRaw(features)
        Vectors.dense(1.0 / (1.0 + math.exp(m)), 1.0 / (1.0 + math.exp(-m)))
    }

    protected def probability2prediction(value: Vector): Double = {
        if (value(1) >= $(threshold)) {
            1.0
        } else {
            0.0
        }
    }

    protected def raw2prediction(features: Vector): Double = {
        val v = predictionRaw(features)
        if (1.0 / (1.0 + math.exp(-v)) >= $(threshold)) {
            1.0
        } else {
            0.0
        }
    }

    /**
      * If the prediction column or probability column is set returns the current model and prediction column,
      * otherwise generates a new column and sets it as the prediction column on a new copy
      * of the current model.
      */
    private[regression] def findBinarySummaryModel: (FMModel, String, String) = {
        val model = if ($(probabilityCol).isEmpty && $(predictionCol).isEmpty) {
            copy(ParamMap.empty)
                    .setProbabilityCol("probability_" + java.util.UUID.randomUUID.toString)
                    .setPredictionCol("prediction_" + java.util.UUID.randomUUID.toString)
        } else if ($(probabilityCol).isEmpty) {
            copy(ParamMap.empty).setProbabilityCol("probability_" + java.util.UUID.randomUUID.toString)
        } else if ($(predictionCol).isEmpty) {
            copy(ParamMap.empty).setPredictionCol("prediction_" + java.util.UUID.randomUUID.toString)
        } else {
            this
        }
        (model, model.getProbabilityCol, model.getPredictionCol)
    }
    /**
      * If the prediction column is set returns the current model and prediction column,
      * otherwise generates a new column and sets it as the prediction column on a new copy
      * of the current model.
      */
    private[regression] def findRegressionSummaryModel: (FMModel, String) = {
        val model = if ($(predictionCol).isEmpty) {
            copy(ParamMap.empty).setPredictionCol("prediction_" + java.util.UUID.randomUUID.toString)
        } else {
            this
        }
        (model, model.getPredictionCol)
    }

    override def copy(extra: ParamMap): FMModel = {
        val model = copyValues(new FMModel(uid, numFeatures, latentVectors, biases,
            linearCoefficients, interceptCoefficient), extra)
        model.setSummary(trainingSummary).setParent(parent)
    }

    // not save parent and training summary
    def write: MLWriter = new FMModel.FMModelWriter(this)

    override protected def predict(features: Vector): Double = {
        if ($(forBinaryClassification)) {
            raw2prediction(features)
        } else {
            predictionRaw(features)
        }
    }
}

object FMModel extends MLReadable[FMModel] {
    override def load(path: String): FMModel = super.load(path)

    //    val a = Pipeline.read.load
    override def read: MLReader[FMModel] = new FMModelReader()

    private[FMModel] class FMModelWriter(model: FMModel) extends MLWriter with Logging {
        private case class Data(
                               numFeatures: Int,
                               latentVectors: Array[Double],
                               biases: Array[Double],
                               linearCoefficients: Array[Double],
                               interceptCoefficient: Double
                               )
        override protected def saveImpl(path: String): Unit = {
            // Save metadata and Params
            DefaultParamsWriter.saveMetadata(model, path, sc)
            // Save model data: intercept, coefficients, scale
            val data = Data(model.numFeatures, model.latentVectors, model.biases, model.linearCoefficients, model.interceptCoefficient)
            val dataPath = new Path(path, "data").toString
            sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
        }
    }
    private class FMModelReader extends MLReader[FMModel] {

        /** Checked against metadata when loading model */
        private val className = classOf[FMModel].getName

        override def load(path: String): FMModel = {
            val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
            val (major, minor) = VersionUtils.majorMinorVersion(metadata.sparkVersion)

            val dataPath = new Path(path, "data").toString
            val data = sparkSession.read.format("parquet").load(dataPath)

            val model = if (major.toInt < 2 || (major.toInt == 2 && minor.toInt < 3)) {
                // before 2.3
                // TODO
                throw new SparkException("Only support spark version 2.3")
            } else {
                // 2.3+
                val Row(numFeatures: Int, latentVectors: Array[Double], biases: Array[Double], linearCoefficients: Array[Double], interceptCoefficient: Double) = data
                        .select("numFeatures", "latentVectors", "biases", "linearCoefficients",
                            "interceptCoefficient", "interceptCoefficient").head()
                new FMModel(metadata.uid, numFeatures, latentVectors, biases, linearCoefficients, interceptCoefficient)
            }

            DefaultParamsReader.getAndSetParams(model, metadata)
            model
        }
    }

}

// utility for summarizing
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

        if (label == 1.0) {
            positiveClass = (positiveClass._1 + 1, positiveClass._2 + weight)
            this
        } else if (label == 0.0) {
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

sealed trait FMSummary extends Serializable {

    /**
      * Dataframe output by the model's `transform` method.
      */
    def predictions: DataFrame

    /** Field in "predictions" which gives the prediction of class. */
    def predictionCol: String

    /** Field in "predictions" which gives the true label of each instance (if available). */
    def labelCol: String

    /** Field in "predictions" which gives the features of each instance as a vector. */
    def featuresCol: String
}

sealed trait FMTrainingSummary extends FMSummary {

    def objectiveHistory: Array[Double]

    def totalIterations: Int = objectiveHistory.length
}

// most code of this class is copy from spark LogisticRegressionSummary and  BinaryLogisticRegressionSummary
sealed trait BinaryFMSummary extends FMSummary {

    private val sparkSession = predictions.sparkSession
    import sparkSession.implicits._

    /**
      * Returns the receiver operating characteristic (ROC) curve,
      * which is a Dataframe having two fields (FPR, TPR)
      * with (0.0, 0.0) prepended and (1.0, 1.0) appended to it.
      * See http://en.wikipedia.org/wiki/Receiver_operating_characteristic
      *
      * @note This ignores instance weights (setting all to 1.0) from `LogisticRegression.weightCol`.
      *       This will change in later Spark versions.
      */
    @transient lazy val roc: DataFrame = binaryMetrics.roc().toDF("FPR", "TPR")



    /**
      * Computes the area under the receiver operating characteristic (ROC) curve.
      *
      * @note This ignores instance weights (setting all to 1.0) from `LogisticRegression.weightCol`.
      *       This will change in later Spark versions.
      */
    lazy val areaUnderROC: Double = binaryMetrics.areaUnderROC()
    /**
      * Returns the precision-recall curve, which is a Dataframe containing
      * two fields recall, precision with (0.0, 1.0) prepended to it.
      *
      * @note This ignores instance weights (setting all to 1.0) from `LogisticRegression.weightCol`.
      *       This will change in later Spark versions.
      */
    @transient lazy val pr: DataFrame = binaryMetrics.pr().toDF("recall", "precision")
    /**
      * Returns a dataframe with two fields (threshold, F-Measure) curve with beta = 1.0.
      *
      * @note This ignores instance weights (setting all to 1.0) from `LogisticRegression.weightCol`.
      *       This will change in later Spark versions.
      */
    @transient lazy val fMeasureByThreshold: DataFrame = {
        binaryMetrics.fMeasureByThreshold().toDF("threshold", "F-Measure")
    }
    /**
      * Returns a dataframe with two fields (threshold, precision) curve.
      * Every possible probability obtained in transforming the dataset are used
      * as thresholds used in calculating the precision.
      *
      * @note This ignores instance weights (setting all to 1.0) from `LogisticRegression.weightCol`.
      *       This will change in later Spark versions.
      */
    @transient lazy val precisionByThreshold: DataFrame = {
        binaryMetrics.precisionByThreshold().toDF("threshold", "precision")
    }
    /**
      * Returns a dataframe with two fields (threshold, recall) curve.
      * Every possible probability obtained in transforming the dataset are used
      * as thresholds used in calculating the recall.
      *
      * @note This ignores instance weights (setting all to 1.0) from `LogisticRegression.weightCol`.
      *       This will change in later Spark versions.
      */
    @transient lazy val recallByThreshold: DataFrame = {
        binaryMetrics.recallByThreshold().toDF("threshold", "recall")
    }
    /**
      * Returns confusion matrix:
      * the first column (resp. row) means predicted (resp. true) label is positive
      * while the second is negative.
      */
    lazy val confusionMatrix: Matrix = {
        val m = binaryclassMetrics.confusionMatrix
        Matrices.dense(2, 2, Array(m(1, 1), m(0, 1), m(1, 0), m(0, 0)))

    }

    @transient private val binaryclassMetrics = {
        new MulticlassMetrics(
            predictions.select(
                col(predictionCol),
                col(labelCol).cast(DoubleType))
                    .rdd.map { case Row(prediction: Double, label: Double) => (prediction, label) })
    }
    // BinaryClassificationMetrics. For now the default is set to 100.
    @transient private val binaryMetrics = new BinaryClassificationMetrics(
        predictions.select(col(probabilityCol), col(labelCol)).rdd.map {
            case Row(score: Vector, label: Double) => (score(1), label)
        }, bin
    )

    /** Field in "predictions" which gives the probability of each label as a vector. */
    def probabilityCol: String

    def bin: Int

    /** precision of positive */
    def precision: Double = precision(1.0)

    /** precision of class. Note 1.0 is positive while 0.0 is negative. */
    def precision(label: Double): Double = binaryclassMetrics.precision(label)

    /** (positive, negative) precision */
    def precisions: (Double, Double) = (precision(1.0), precision(0.0))

    /** recall of positive */
    def recall: Double = recall(1.0)

    /** recall of class. Note 1.0 is positive while 0.0 is negative. */
    def recall(label: Double): Double = binaryclassMetrics.recall(label)

    /** (positive, negative) recall */
    def recalls: (Double, Double) = (recall(1.0), recall(0.0))

    /**
      * Returns accuracy.
      * (equals to the total number of correctly classified instances
      * out of the total number of instances.)
      */
    def accuracy: Double = binaryclassMetrics.accuracy

    /** Returns true positive count. */

    def truePositive: Double = confusionMatrix(0, 0)

    /** Returns false positive count. */
    def falsePositive: Double = binaryclassMetrics.confusionMatrix(0, 1)

    /** Returns true positive count. */

    def trueNegative: Double = binaryclassMetrics.confusionMatrix(1, 1)

    /** Returns false positive count. */
    def falseNegative: Double = binaryclassMetrics.confusionMatrix(1, 0)

    /** Returns f1-measure for each label (positive, negative). */
    def fMeasure: (Double, Double) = fMeasure(1.0)

    /** Returns f-measure for each label (positive, negative). */
    def fMeasure(beta: Double): (Double, Double) = {
        val t = binaryclassMetrics.labels.map(label => binaryclassMetrics.fMeasure(label, beta))
        (t(1), t(0))
    }

}

sealed trait BinaryFMTrainingSummary extends BinaryFMSummary with FMTrainingSummary

/** summary of FM for regression */

sealed trait RegressionFMSummary extends FMSummary {
    /** Residuals (label - predicted value) */
    @transient lazy val residuals: DataFrame = {
        val t = udf { (pred: Double, label: Double) => label - pred }
        predictions.select(t(col(predictionCol), col(labelCol)).as("residuals"))
    }
    /** Number of instances in DataFrame predictions */
    lazy val numInstances: Long = predictions.count()
    /**
      * The weighted residuals, the usual residuals rescaled by
      * the square root of the instance weights.
      */
    lazy val devianceResiduals: Array[Double] = {
        val weighted =
            if (!privateModel.isDefined(privateModel.weightCol) || privateModel.getWeightCol.isEmpty) {
                lit(1.0)
            } else {
                sqrt(col(privateModel.getWeightCol))
            }
        val dr = predictions
                .select(col(privateModel.getLabelCol).minus(col(privateModel.getPredictionCol))
                        .multiply(weighted).as("weightedResiduals"))
                .select(min(col("weightedResiduals")).as("min"), max(col("weightedResiduals")).as("max"))
                .first()
        Array(dr.getDouble(0), dr.getDouble(1))
    }
    @transient private lazy val predictionAndObservations = {
        predictions.select(col(predictionCol), col(labelCol).cast(DoubleType))
                .rdd
                .map { case Row(pred: Double, label: Double) => (pred, label) }
    }
    // Multivariate Statistical Summary
    @transient private lazy val summary = {
        val summary: MultivariateStatisticalSummary = predictionAndObservations
                .map { case (pred, label) => Vectors.dense(label, label - pred) }
                .treeAggregate(new MultivariateOnlineSummarizer())(
                    (summary, v) => summary.add(v), (sum1, sum2) => sum1.merge(sum2)
                )
        summary
    }
    private lazy val SSerr = math.pow(summary.normL2(1), 2)
    private lazy val SStot = summary.variance(0) * (summary.count - 1)
    private lazy val SSreg = {
        val yMean = summary.mean(0)
        predictionAndObservations.map {
            case (prediction, _) => math.pow(prediction - yMean, 2)
        }.sum()
    }

    protected def privateModel: FMModel

    /**
      * Returns the explained variance regression score.
      * explainedVariance = 1 - variance(y - \hat{y}) / variance(y)
      * Reference: <a href="http://en.wikipedia.org/wiki/Explained_variation">
      * Wikipedia explain variation</a>
      */
    def explainedVariance: Double = {
        SSreg / summary.count
    }

    /**
      * Returns the mean absolute error, which is a risk function corresponding to the
      * expected value of the absolute error loss or l1-norm loss.
      *
      * @note This ignores instance weights (setting all to 1.0) from `LinearRegression.weightCol`.
      *       This will change in later Spark versions.
      */
    def meanAbsoluteError: Double = {
        summary.normL1(1) / summary.count
    }

    /**
      * Returns the root mean squared error, which is defined as the square root of
      * the mean squared error.
      *
      * @note This ignores instance weights (setting all to 1.0) from weightCol.
      *       This will change in later Spark versions.
      */
    def rootMeanSquaredError: Double = {
        math.sqrt(this.meanSquaredError)
    }

    /**
      * Returns the mean squared error, which is a risk function corresponding to the
      * expected value of the squared error loss or quadratic loss.
      *
      */
    def meanSquaredError: Double = {
        SSerr / summary.count
    }

    /**
      * Returns R^2^, the coefficient of determination.
      * Reference: <a href="http://en.wikipedia.org/wiki/Coefficient_of_determination">
      * Wikipedia coefficient of determination</a>
      *
      * @note This ignores instance weights (setting all to 1.0) from weightCol.
      *       This will change in later Spark versions.
      */
    def r2: Double = {
        1 - SSerr / SStot
    }

}

sealed trait RegressionFMTrainingSummary extends FMTrainingSummary


private class BinaryFMSummaryImpl private[regression](
                                                             @transient override val predictions: DataFrame,
                                                             override val probabilityCol: String,
                                                             override val predictionCol: String,
                                                             override val labelCol: String,
                                                             override val featuresCol: String,
                                                             override val bin: Int
                                                     )
        extends BinaryFMSummary

private class BinaryFMTrainingSummaryImpl private[regression](
                                                                     predictions: DataFrame,
                                                                     probabilityCol: String,
                                                                     predictionCol: String,
                                                                     labelCol: String,
                                                                     featuresCol: String,
                                                                     bin: Int,
                                                                     override val objectiveHistory: Array[Double]
                                                             )
        extends BinaryFMSummaryImpl(predictions, probabilityCol, predictionCol, labelCol, featuresCol, bin)
                with BinaryFMTrainingSummary

//regression results evaluated on a dataset.
/**
  * ref:
  **/
private class RegressionFMSummaryImpl private[regression](
                                                                 @transient override val predictions: DataFrame,
                                                                 override val predictionCol: String,
                                                                 override val labelCol: String,
                                                                 override val featuresCol: String,
                                                                 override protected val privateModel: FMModel
                                                         ) extends RegressionFMSummary


private class RegressionFMTrainingSummaryImpl private[regression](
                                                                           predictions: DataFrame,
                                                                           predictionCol: String,
                                                                           labelCol: String,
                                                                           featuresCol: String,
                                                                           privateModel: FMModel,
                                                                           override val objectiveHistory: Array[Double]
                                                                   )
        extends RegressionFMSummaryImpl(predictions, predictionCol, labelCol, featuresCol, privateModel)
        with FMTrainingSummary