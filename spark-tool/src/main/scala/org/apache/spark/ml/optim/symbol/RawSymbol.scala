package org.apache.spark.ml.optim.symbol

trait RawSymbol[T] extends Serializable {

    def getValue: T

    val a: States.States = States.Constant

}

object States extends Enumeration {
    type States = Value
    val Trainable = Value("hi") {
        def f = 1
    }

    val Constant = Value("Hello") {
        def f = 2
    }
}
