package org.apache.spark.ml.optim.symbol

class Variable[T <: Serializable] (
                                  init: Int => T
                                  )
        extends RawSymbol[T] with Serializable {

    private val t: T = ???

    override def getValue: T = t
}
