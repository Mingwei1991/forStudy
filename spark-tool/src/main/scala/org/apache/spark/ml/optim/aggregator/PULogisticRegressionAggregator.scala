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

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}

import scala.collection.mutable

private[ml] class PULogisticRegressionAggregator(override val epsilon: Double,
                                                 override val multiplier: Double,
                                                 override val bcFeaturesStd: Broadcast[Array[Double]],
                                                 val fitIntercept: Boolean
                                                )(bcCoefficients: Broadcast[Vector])
  extends LogisticLossPUAggregator {

  private val numFeatures = bcFeaturesStd.value.length

  private val numFeaturesPlusIntercept = if (fitIntercept) numFeatures + 1 else numFeatures

  override protected val dim: Int = bcCoefficients.value.size

  require(dim == numFeaturesPlusIntercept, s"Expected $numFeaturesPlusIntercept " +
    s"coefficients but got $dim")

  @transient private lazy val coefficientsArray: Array[Double] = bcCoefficients.value match {
    case DenseVector(values) => values
    case _ => throw new IllegalArgumentException(s"coefficients only supports dense vector but " +
      s"got type ${bcCoefficients.value.getClass}.)")
  }

  // todo handle
  override protected def marginAndGradient(features: Vector): (Double, Vector) = {
    val localFeaturesStd = bcFeaturesStd.value
    val localCoefficients = coefficientsArray
    val indexBuilder = mutable.ArrayBuilder.make[Int]
    val gradientBuilder = mutable.ArrayBuilder.make[Double]

    var sum = 0.0
    features.foreachActive { (index, value) =>
      if (localFeaturesStd(index) != 0.0 && value != 0.0) {
        sum += localCoefficients(index) * value / localFeaturesStd(index)
        indexBuilder += index
        gradientBuilder += value / localFeaturesStd(index)
      }
    }
    if (fitIntercept) {
      sum += localCoefficients(numFeatures)
      indexBuilder += numFeatures
      gradientBuilder += 1
    }
    (sum, Vectors.sparse(dim, indexBuilder.result, gradientBuilder.result))
  }
}
