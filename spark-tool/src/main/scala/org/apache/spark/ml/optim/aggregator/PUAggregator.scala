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

package org.apache.spark.ml.optim.aggregator

import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.optim.aggregator.{DifferentiableLossAggregator => DLA}

// TODO
/**
  * Only support linear additional term for labeled positive samples
  **/
private[ml] abstract class PUAggregator[Agg <: PUAggregator[Agg]]
  extends DLA[Instance, Agg] {
  self: Agg =>

  // Epsilon parameter in the epsilon-insensitive loss function for unlabeled samples
  val epsilon: Double

  def add(instance: Instance): Agg = {
    val Instance(label, weight, features) = instance
    require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")

    if (weight == 0.0) return this

    updateInPlace(features, weight, label)
    weightSum += weight
    this
  }

  private def updateInPlace(features: Vector, weight: Double, label: Double): Unit = {
    val localGradientSumArray = this.gradientSumArray
    val (raw, rawGrad) = rawPredictionAndGradient(features)

    val loss = raw2loss(raw)

    if (label != 0.0 || loss > epsilon) {
      lossSum += weight * loss

      var rate = weight * raw2lossGradient(raw)
      rawGrad.foreachActive { (index, value) =>
        localGradientSumArray(index) += rate * value
      }
      // additional update for labeled positive samples
      if (label == 1.0) {
        rate = additionalRate * weight
        lossSum += rate * raw

        rawGrad.foreachActive { (index, value) =>
          if (value != 0.0) {
            localGradientSumArray(index) += rate * value
          }
        }
      }
    }
  }

  protected def raw2loss(raw: Double): Double

  protected def raw2lossGradient(raw: Double): Double

  // rate of additional term for labeled positive samples
  protected def additionalRate: Double

  protected def rawPredictionAndGradient(features: Vector): (Double, Vector)
}
