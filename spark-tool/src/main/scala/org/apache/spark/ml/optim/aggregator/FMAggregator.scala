package org.apache.spark.ml.optim.aggregator

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.optim.aggregator.{DifferentiableLossAggregator => DLA}

private[ml] abstract class FMAggregatorBase[A <: DLA[Instance, A]](
                                                                          bcFeaturesStd: Broadcast[Array[Double]],
                                                                          fitIntercept: Boolean,
                                                                          fitLinear: Boolean,
                                                                          vectorSize: Int,
                                                                          fitBias: Boolean
                                                                  )
                                                                  (bcCoefficients: Broadcast[Vector])
        extends DLA[Instance, A] {
    self: A => // enforce classes that extend this to be the same type as `A`

    private val numFeatures = bcFeaturesStd.value.length
    private val totalSize: Int = {
        var s = vectorSize * totalSize
        if (fitIntercept) s += 1
        if (fitLinear) s += numFeatures
        if (fitBias) s += numFeatures
        s
    }

    private val coefficientSize =bcCoefficients.value.size

    protected override val dim: Int = coefficientSize

    require(coefficientSize == totalSize, s"Expected $totalSize " +
            s"coefficients but got $coefficientSize")

    @transient private lazy val coefficientsArray: Array[Double] = bcCoefficients.value match {
        case DenseVector(values) => values
        case _ => throw new IllegalArgumentException(s"coefficients only supports dense vector but " +
                s"got type ${bcCoefficients.value.getClass}.)")
    }

    /**
      * gradient of loss function at value = error
      * */
    def lossGradient(error: Double): Double

}

private[ml] class FMAggregator {

}

