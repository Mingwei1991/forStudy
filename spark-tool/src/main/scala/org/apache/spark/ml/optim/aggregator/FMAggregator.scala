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
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.optim.aggregator.{DifferentiableLossAggregator => DLA}
import org.apache.spark.mllib.util.MLUtils

private[ml] abstract class FMAggregatorBase[A <: DLA[Instance, A]](
                                                                          numFeatures: Int,
                                                                          vectorSize: Int,
                                                                          fitBias: Boolean,
                                                                          fitLinear: Boolean,
                                                                          fitIntercept: Boolean
                                                                  )
                                                                  (bcCoefficients: Broadcast[Vector])
        extends DLA[Instance, A] {
    self: A => // enforce classes that extend this to be the same type as `A`

    @transient private lazy val coefficientsArray: Array[Double] = bcCoefficients.value match {
        case DenseVector(values) => values
        case _ => throw new IllegalArgumentException(s"coefficients only supports dense vector but " +
                s"got type ${bcCoefficients.value.getClass}.)")
    }
    protected override val dim: Int = bcCoefficients.value.size

    private val totalSize: Int = {
        var s = vectorSize * numFeatures
        if (fitIntercept) s += 1
        if (fitLinear) s += numFeatures
        if (fitBias) s += numFeatures
        s
    }

    require(dim == totalSize, s"Expected $totalSize " +
            s"coefficients but got $dim")

    /**
      * gradient of loss function at value = error
      **/
    def lossGradient(combinationSum: Double, label: Double): Double

    def loss(combinationSum: Double, label: Double): Double

    def add(instance: Instance): A = {
        val Instance(label, weight, features) = instance
        require(numFeatures == features.size, s"Dimensions mismatch when adding new instance." +
                s" Expecting $numFeatures but got ${features.size}.")
        require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")

        if (weight == 0.0) return this

        updateInPlace(features, weight, label)
        weightSum += weight

        this
    }

    def updateInPlace(features: Vector, weight: Double, label: Double): Unit = {
        // reduce cost of visit array data
        val localCoefficients = coefficientsArray
        val localGradientArray = gradientSumArray
        // compute combination sum
        var combinationSum = 0.0
        var originPosition = 0
        // combination gradient
        val combinationGradient = new Array[Double](dim)
        // reuse index i
        var i = 0

        val vectorSum = new Array[Double](vectorSize)
        features.foreachActive { (index, value) =>
            if (value != 0.0) {
                originPosition = index * vectorSize
                var norm2 = 0.0
                while (i < vectorSize) {
                    val coefficientPosition = originPosition + i
                    vectorSum(i) += value * localCoefficients(coefficientPosition)
                    norm2 += localCoefficients(coefficientPosition) * localCoefficients(coefficientPosition)
                    combinationGradient(coefficientPosition) -= value * value * localCoefficients(coefficientPosition)
                    i += 1
                }
                combinationSum -= value * value * norm2
                i = 0
            }
        }

        while (i < vectorSize) {
            combinationSum += vectorSum(i) * vectorSum(i)
            i += 1
        }
        i = 0

        combinationSum /= 2.0
        // calculate latent vector gradient
        features.foreachActive { (index, value) =>
            if (value != 0.0) {
                val p = index * vectorSize
                while (i < vectorSize) {
                    combinationGradient(p + i) += value * vectorSum(i)
                    i += 1
                }
                i = 0
            }
        }

        originPosition = numFeatures * vectorSize

        if (fitBias) {
            var sum = 0.0
            features.foreachActive { (index, value) =>
                if (value != 0.0) {
                    val t = localCoefficients(originPosition + index) * value
                    sum += t
                    combinationGradient(originPosition + index) -= t * value
                    combinationSum -= t * t / 2.0
                }

            }

            combinationSum += sum * sum / 2
            // calculate gradient
            features.foreachActive { (index, value) =>
                combinationGradient(originPosition + index) += sum * value
            }
            originPosition += numFeatures
        }

        if (fitLinear) {
            features.foreachActive { (index, value) =>
                if (value != 0.0) {
                    combinationSum += localCoefficients(originPosition + index) * value
                    combinationGradient(originPosition + index) += value
                }
            }
        }

        if (fitIntercept) {
            combinationSum += localCoefficients(dim - 1)
            combinationGradient(dim - 1) += 1
        }
        // update loss
        lossSum += weight * loss(combinationSum, label)

        val multiplier = weight * lossGradient(combinationSum, label)
        while (i < dim) {
            localGradientArray(i) += multiplier * combinationGradient(i)
            i += 1
        }


    }

}

private[ml] abstract class MarginFMAggregator[A <: FMAggregatorBase[A]](
                                                                               numFeatures: Int,
                                                                               vectorSize: Int,
                                                                               fitBias: Boolean,
                                                                               fitLinear: Boolean,
                                                                               fitIntercept: Boolean
                                                                       )
                                                                       (bcCoefficients: Broadcast[Vector])
        extends FMAggregatorBase[A](numFeatures, vectorSize, fitBias,  fitLinear, fitIntercept)(bcCoefficients) {
    self: A =>
    /**
      * margin type loss
      **/
    def loss(margin: Double): Double

    def lossGradient(margin: Double): Double

    override def lossGradient(combinationSum: Double, label: Double): Double = lossGradient(combinationSum - label)

    override def loss(combinationSum: Double, label: Double): Double = loss(combinationSum - label)
}

private[ml] class LeastSquaresFMAggregator(
                                                  numFeatures: Int,
                                                  vectorSize: Int,
                                                  fitBias: Boolean,
                                                  fitLinear: Boolean,
                                                  fitIntercept: Boolean
                                          )
                                          (bcCoefficients: Broadcast[Vector])
        extends MarginFMAggregator[LeastSquaresFMAggregator](numFeatures, vectorSize, fitBias,  fitLinear, fitIntercept)(bcCoefficients) {
    /**
      * margin type loss
      **/
    override def loss(margin: Double): Double = margin * margin / 2

    override def lossGradient(margin: Double): Double = margin
}

private[ml] class LogisticFMAggregator(
                                              numFeatures: Int,
                                              vectorSize: Int,
                                              fitBias: Boolean,
                                              fitLinear: Boolean,
                                              fitIntercept: Boolean
                                      )
                                      (bcCoefficients: Broadcast[Vector])
        extends FMAggregatorBase[LogisticFMAggregator](numFeatures, vectorSize, fitBias,  fitLinear, fitIntercept)(bcCoefficients) {
    /**
      * logistic type loss
      **/
    /**
      * gradient of loss function at value = error
      **/
    override def lossGradient(combinationSum: Double, label: Double): Double = 1.0 / (1.0 + math.exp(-combinationSum)) - label


    override def loss(combinationSum: Double, label: Double): Double = {
        if (label > 0.0) {
             MLUtils.log1pExp(-combinationSum)
        }
        else {
             MLUtils.log1pExp(combinationSum)
        }
    }
}




