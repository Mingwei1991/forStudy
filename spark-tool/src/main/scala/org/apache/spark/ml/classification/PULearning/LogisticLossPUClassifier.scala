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

package org.apache.spark.ml.classification.PULearning

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{LBFGS => BreezeLBFGS, OWLQN => BreezeOWLQN}
import org.apache.spark.SparkException
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{Matrix, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.optim.aggregator.LogisticLossPUAggregator
import org.apache.spark.ml.optim.loss.{L2Regularization, RDDLossFunction}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param._
import org.apache.spark.ml.tool.BinaryClassSummarizer
import org.apache.spark.ml.util.{Instrumentation, SchemaUtils}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{DataType, DoubleType, StructType}
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable

// TODO set bounds on coefficients and  bound constrained optimization
// TODO fit stds of features
private[classification] trait LogisticLossPUClassifierParams extends PUClassifierParams with HasProbabilityCol with HasTol
  with HasRegParam with HasElasticNetParam with HasMaxIter with HasWeightCol with HasAggregationDepth {

  final val memory = new IntParam(this, "memory", "memory for L-BFGS and OWLQN", ParamValidators.gt(0))

  def getMemory: Int = $(memory)

  override protected def validateAndTransformSchema(
                                                     schema: StructType,
                                                     fitting: Boolean,
                                                     featuresDataType: DataType): StructType = {
    val parentSchema = super.validateAndTransformSchema(schema, fitting, featuresDataType)
    SchemaUtils.appendColumn(parentSchema, $(probabilityCol), new VectorUDT)
  }
}

abstract class LogisticLossPUClassifier[E <: LogisticLossPUClassifier[E, M], M <: LogisticLossPUClassifierModel[M]]
  extends PUClassifier[E, M] with LogisticLossPUClassifierParams {

  /** @group setParam */
  def setProbabilityCol(value: String): E = set(probabilityCol, value).asInstanceOf[E]

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

  def setMemory(value: Int): this.type = set(memory, value)

  setDefault(memory -> 10)

  /**
    * Set the convergence tolerance of iterations.
    * Smaller value will lead to higher accuracy at the cost of more iterations.
    * Default is 1E-6.
    *
    * @group setParam
    */
  def setTol(value: Double): this.type = set(tol, value)

  setDefault(tol -> 1E-6)


  // default threshold
  setDefault(threshold -> 0.5)

  // default epsilon
  setDefault(epsilon -> 0.0)

  /**
    * Sets the value of param [[weightCol]].
    * If this is not set or empty, we treat all instance weights as 1.0.
    * Default is not set, so all instances have weight one.
    *
    * @group setParam
    */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /**
    * Suggested depth for treeAggregate (greater than or equal to 2).
    * If the dimensions of features or the number of partitions are large,
    * this param could be adjusted to a larger size.
    * Default is 2.
    *
    * @group setParam
    */
  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)

  setDefault(aggregationDepth -> 2)

  override def copy(extra: ParamMap): E = defaultCopy(extra)


  override protected[spark] def train(dataset: Dataset[_]): M = {
    val handlePersistence = dataset.storageLevel == StorageLevel.NONE
    train(dataset, handlePersistence)
  }


  protected[spark] def train(
                              dataset: Dataset[_],
                              handlePersistence: Boolean): M = {

    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[Instance] =
      dataset.select(col($(labelCol)), w, col($(featuresCol))).rdd.map {
        case Row(label: Double, weight: Double, features: Vector) =>
          Instance(label, weight, features)
      }
    // persis model
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)
    val instr = Instrumentation.create(this, instances)
    // TODO, not sure this is necessary
    instr.logParams(logParams: _*)

    val (featureSummarizer, labelSummarizer) = {
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
    // TODO handle invalid
    val numInvalid = labelSummarizer.countInvalid
    // require numInvalid=0
    if (numInvalid != 0) {
      val msg = s"Classification labels should be in 0.0 or 1.0. " +
        s"Found $numInvalid invalid labels."
      logError(msg)
      throw new SparkException(msg)
    }

    val histogram = labelSummarizer.histogram // length = 2
    val positiveWeight = histogram(1)._2
    val unlabeledWeight = histogram(0)._2
    val multiplier = (positiveWeight + unlabeledWeight) * getPi / positiveWeight
    require(positiveWeight != 0.0, s"The sum of weights of positive labeled class must be greater than 0.")
    require(unlabeledWeight != 0.0, s"The sum of the weights of unlabeled samples is zero.")


    instr.logNamedValue("sample size", histogram(1)._1)

    val numFeatures: Int = instances.first().features.size
    instr.logNumFeatures(numFeatures)

    val featuresStd = featureSummarizer.variance.toArray.map(math.sqrt)

    val numNonConstantFeatures = featuresStd.count(_ == 0.0)
    if (numNonConstantFeatures != 0)
      logWarning(s"There are $numNonConstantFeatures features with constant values.")

    val initCoefficients: Vector = initModel(numFeatures)
    val numCoefficients = initCoefficients.size
    instr.logNamedValue("coefficients size", numCoefficients)

    // regularization parameter
    val regParamL1 = $(elasticNetParam) * $(regParam)

    val bcFeaturesStd = instances.context.broadcast(featuresStd)
    val getAggregatorFunc: Broadcast[Vector] => LogisticLossPUAggregator = getAggregatorFunction(multiplier, bcFeaturesStd)
    val costFun = new RDDLossFunction(instances, getAggregatorFunc, l2Regularization(featuresStd), $(aggregationDepth))

    // select optimize according to L1 regularization
    // The memory used in L-BFGS or OWLQN
    val optimizer = if (regParamL1 == 0.0) {
      new BreezeLBFGS[BDV[Double]]($(maxIter), getMemory, $(tol))
    } else {
      new BreezeOWLQN[Int, BDV[Double]]($(maxIter), getMemory, regParamL1Fun(featuresStd), $(tol))
    }

    val states = optimizer.iterations(costFun, initCoefficients.asBreeze.toDenseVector)


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
    if (handlePersistence) instances.unpersist()
    bcFeaturesStd.destroy()
    0 until numFeatures foreach { index =>
      if (featuresStd(index) != 0.0) parameters(index) = parameters(index) / featuresStd(index)
    }

    copyValues(constructModel(parameters)).setParent(this).setSummary(arrayBuilder.result, 10)
  }

  protected def initModel(numFeatures: Int): Vector

  protected def constructModel(coefficients: Array[Double]): M

  protected def logParams: Seq[Param[_]]

  protected def getAggregatorFunction(multiplier: Double, bcFeaturesStd: Broadcast[Array[Double]]): Broadcast[Vector] => LogisticLossPUAggregator

  protected def regParamL1Fun(featuresStd: Array[Double]): Int => Double

  protected def l2Regularization(featuresStd: Array[Double]): Option[L2Regularization]
}

object LogisticLossPUClassifier {

}

// TODO
abstract class LogisticLossPUClassifierModel[M <: LogisticLossPUClassifierModel[M]]
  extends PUClassifierModel[M] with LogisticLossPUClassifierParams {

  private[classification] def setSummary(trainSummary: Array[Double], bin: Int): this.type

  def hasSummary: Boolean

  def setProbabilityCol(value: String): this.type

  override def transform(dataset: Dataset[_]): DataFrame = {
    var outputData = dataset
    var numColsOutput = 0
    if (getRawPredictionCol.nonEmpty) {
      val predictRawUDF: Column = udf(predictRaw _).apply(col(getFeaturesCol))
      outputData = outputData.withColumn(getRawPredictionCol, predictRawUDF)
      numColsOutput += 1
    }
    if (getProbabilityCol.nonEmpty) {
      val probabilityCol = if (getRawPredictionCol.nonEmpty) {
        udf(raw2probability _).apply(col(getRawPredictionCol))
      } else {
        udf(raw2probability _).apply(col(getFeaturesCol))
      }
      outputData = outputData.withColumn(getProbabilityCol, probabilityCol)
      numColsOutput += 1
    }
    if (getPredictionCol.nonEmpty) {
      val predictionCol = if (getProbabilityCol.nonEmpty) {
        udf(probability2prediction _).apply(col(getProbabilityCol))
      } else if (getRawPredictionCol.nonEmpty) {
        udf(raw2prediction _).apply(col(getRawPredictionCol))
      } else {
        udf(predict _).apply(col(getFeaturesCol))
      }
      outputData = outputData.withColumn(getPredictionCol, predictionCol)
    }
    require(numColsOutput > 0, s"$uid: LogisticLossPUClassifier.transform() was called as NOOP" +
      " since no output columns were set.")
    outputData.toDF
  }

  /**
    * As the compute is under control, we do not need  raw2probabilityInPlace like ProbabilisticClassifier
    **/
  def raw2probability(value: Vector): Vector = {
    val p = 1 / (1 + math.exp(-value(1)))
    Vectors.dense(1 - p, p)
  }

  def probability2prediction(value: Vector): Double = {
    if (value(1) > getThreshold) {
      1.0
    } else {
      0.0
    }
  }

  override def raw2prediction(rawPrediction: Vector): Double = {
    val t = getThreshold
    val rawThreshold = if (t == 0.0) {
      Double.NegativeInfinity
    } else if (t == 1.0) {
      Double.PositiveInfinity
    } else {
      math.log(t / (1.0 - t))
    }
    if (rawPrediction(1) > rawThreshold) 1 else 0
  }
}

private[ml] object LogisticLossPUClassifierModel {

}

// todo
sealed trait LogisticLossPUClassifierSummary extends Serializable {

}

// todo
sealed trait LogisticLossPUClassifierSummaryImp extends Serializable {
  /**
    * Dataframe output by the model's `transform` method.
    */
  def predictions: DataFrame

  /** Field in "predictions" which gives the probability of each class as a vector. */
  def probabilityCol: String

  /** Field in "predictions" which gives the prediction of each class. */
  def predictionCol: String

  /** Field in "predictions" which gives the true label of each instance (if available). */
  def labelCol: String

  /** Field in "predictions" which gives the features of each instance as a vector. */
  def featuresCol: String

  @transient private val multiclassMetrics = {
    new MulticlassMetrics(
      predictions.select(
        col(predictionCol),
        col(labelCol).cast(DoubleType))
        .rdd.map { case Row(prediction: Double, label: Double) => (prediction, label) })
  }

  /**
    * Returns confusion matrix:
    * predicted classes are in columns,
    * they are ordered by class label ascending,
    * as in "labels"
    */
  def confusionMatrix: Matrix = multiclassMetrics.confusionMatrix.asML

  /** Returns true positive . */
  def truePositive = confusionMatrix(1, 1)

  /** Returns false positive . */
  def falsePositive = confusionMatrix(0, 1)

  /** Returns true positive . */
  def trueNegative = confusionMatrix(0, 0)

  /** Returns false positive . */
  def falseNegative = confusionMatrix(1, 0)

  /** Return precision for positive category. */
  def positivePrecision: Double = {
    multiclassMetrics.precision(1.0)
  }

  /** Return precision for negative category. */
  def negativePrecision: Double = {
    multiclassMetrics.precision(0.0)
  }

  /** Return recall for positive category. */
  def positiveRecall: Double = {
    multiclassMetrics.recall(1.0)
  }

  /** Return recall for negative category. */
  def negativeRecall: Double = {
    multiclassMetrics.recall(0.0)
  }

  /** Returns true positive rate for each label (category). */
  def trueRateByLabel: Array[Double] = recallByLabel


  /** Returns false positive rate for each label (category). */
  def falseRateByLabel: Array[Double] = {
    multiclassMetrics.labels.map(label => multiclassMetrics.falsePositiveRate(label))
  }

  /** Returns precision for each label (category). */
  def precisionByLabel: Array[Double] = {
    multiclassMetrics.labels.map(label => multiclassMetrics.precision(label))
  }

  /** Returns recall for each label (category). */
  def recallByLabel: Array[Double] = {
    multiclassMetrics.labels.map(label => multiclassMetrics.recall(label))
  }

  /** Returns f-measure for each label (category). */
  def fMeasureByLabel(beta: Double): Array[Double] = {
    multiclassMetrics.labels.map(label => multiclassMetrics.fMeasure(label, beta))
  }

  /** Returns f1-measure for each label (category). */
  def fMeasureByLabel: Array[Double] = fMeasureByLabel(1.0)

  /**
    * Returns accuracy.
    * (equals to the total number of correctly classified instances
    * out of the total number of instances.)
    */
  def accuracy: Double = multiclassMetrics.accuracy
}