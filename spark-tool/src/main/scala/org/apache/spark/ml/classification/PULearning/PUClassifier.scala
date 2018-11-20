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

import org.apache.spark.ml.classification.{ClassificationModel, Classifier, ClassifierParams}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{DoubleParam, ParamValidators}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.{DataType, StructType}

private[classification] trait PUClassifierParams extends ClassifierParams
with HasThreshold {
    /**
      *
      * */
    final val pi = new DoubleParam(this, "pi",
        "the (expect) weighted percentage of positive classes", ParamValidators.inRange(0, 1))

    def getPi: Double = $(pi)

    def setThreshold(value: Double): this.type = set(threshold, value)

    override protected def validateAndTransformSchema(
                                                             schema: StructType,
                                                             fitting: Boolean,
                                                             featuresDataType: DataType): StructType = {
        val parentSchema = super.validateAndTransformSchema(schema, fitting, featuresDataType)
        parentSchema
    }

}

abstract class PUClassifier[E <: PUClassifier[E, M], M <: PUClassifierModel[M]]
        extends Classifier[Vector, E, M] with PUClassifierParams {

    // override method and val from super class involve numClasses
    override final def getNumClasses(dataset: Dataset[_], maxNumClasses: Int): Int = 2



}

abstract class PUClassifierModel[M <: PUClassifierModel[M]]
        extends ClassificationModel[Vector, M] with PUClassifierParams {
    override final def numClasses: Int = 2
}

// todo
trait PUClassifierTrainingSummary extends Serializable
