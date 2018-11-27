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

package org.apache.spark.ml.classification

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.PULearning._
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors}
import org.apache.spark.ml.optim.aggregator.{LogisticLossPUAggregator, PULogisticRegressionAggregator}
import org.apache.spark.ml.optim.loss.L2Regularization
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, MLWritable, MLWriter}

// TODO set bounds on coefficients and  bound constrained optimization
private[classification] trait PULogisticRegressionParams extends LogisticLossPUClassifierParams
  with HasFitIntercept with HasStandardization

// quit like LogisticRegression
class PULogisticRegression(
                            override val uid: String
                          ) extends LogisticLossPUClassifier[PULogisticRegression, PULogisticRegressionModel]
  with PULogisticRegressionParams with DefaultParamsWritable with Logging {
  def this() = this(Identifiable.randomUID("pulogisticregression"))

  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)

  setDefault(fitIntercept, true)


  def setStandardization(value: Boolean): this.type = set(standardization, value)

  override protected def initModel(numFeatures: Int): Vector = {
    if ($(fitIntercept)) {
      Vectors.dense(new Array[Double](numFeatures + 1))
    } else {
      Vectors.dense(new Array[Double](numFeatures))
    }
  }

  override protected def constructModel(coefficients: Array[Double]): PULogisticRegressionModel = {
    if ($(fitIntercept)) {
      val size = coefficients.length
      new PULogisticRegressionModel(uid, Vectors.dense(coefficients.slice(0, size)), coefficients(size - 1))
    } else {
      new PULogisticRegressionModel(uid, Vectors.dense(coefficients), 0.0)
    }
  }

  override protected def logParams: Seq[Param[_]] = Seq(pi, regParam, elasticNetParam, threshold,
    maxIter, tol, fitIntercept)

  override protected def getAggregatorFunction(multiplier: Double, bcFeaturesStd: Broadcast[Array[Double]]): Broadcast[Vector] => LogisticLossPUAggregator = {
    new PULogisticRegressionAggregator(getEpsilon, multiplier, bcFeaturesStd, getFitIntercept)(_)
  }

  override protected def regParamL1Fun(featuresStd: Array[Double]): Int => Double = {
    val regParamL1 = $(elasticNetParam) * $(regParam)
    if ($(standardization)) {
      index: Int => {
        val isIntercept = $(fitIntercept) && index >= featuresStd.length
        if (isIntercept) {
          0.0
        } else {
          regParamL1
        }
      }
    } else {
      index: Int => {
        val isIntercept = $(fitIntercept) && index >= featuresStd.length
        if (isIntercept) {
          0.0
        } else {
          if (featuresStd(index) != 0.0) {
            regParamL1 / featuresStd(index)
          } else {
            0.0
          }
        }
      }
    }
  }


  override protected def l2Regularization(featuresStd: Array[Double]): Option[L2Regularization] = {
    val regParamL2 = (1.0 - $(elasticNetParam)) * $(regParam)
    if (regParamL2 != 0.0) {
      val shouldApply = (idx: Int) => idx >= 0 && idx < featuresStd.length
      Some(new L2Regularization(regParamL2, shouldApply,
        if ($(standardization)) None else Some(featuresStd)))
    } else {
      None
    }
  }
}

object PULogisticRegression {

}

class PULogisticRegressionModel private[spark](
                                                override val uid: String,
                                                val coefficients: Vector,
                                                val intercept: Double
                                              ) extends LogisticLossPUClassifierModel[PULogisticRegressionModel]
  with PULogisticRegressionParams with MLWritable {

  // todo summary
  //override def write: MLWriter = new LogisticRegressionModel.LogisticRegressionModelWriter(this)

  override def write: MLWriter = ???

  override protected def predictRaw(features: Vector): Vector = {
    var margin = intercept
    features.foreachActive { (index, value) =>
      margin += value * coefficients(index)
    }
    Vectors.dense(-margin, margin)
  }

  override def copy(extra: ParamMap): PULogisticRegressionModel = ???

  override private[classification] def setSummary(trainSummary: Array[Double], bin: Int) = {
    this
  }

  override def hasSummary: Boolean = ???

  override def setProbabilityCol(value: String): PULogisticRegressionModel.this.type = ???
}