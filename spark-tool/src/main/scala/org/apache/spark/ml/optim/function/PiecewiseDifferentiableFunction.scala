package org.apache.spark.ml.optim.function

abstract class PiecewiseDifferentiableFunction {
    val a = 1;

    abstract def forward(value: Double): Double

    abstract def f

}
