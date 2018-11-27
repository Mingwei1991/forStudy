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

package org.apache.spark.ml.tool

// utility for summarizing
private[ml] class BinaryClassSummarizer extends Serializable {
  // the first element is the actual number of instances
  // the second element is weighted sum
  private var positiveClass = 0L -> 0.0
  private var negativeClass = 0L -> 0.0
  // invalid label count
  private var totalInvalidCnt: Long = 0L


  def add(label: Double, weight: Double = 1.0): this.type = {
    require(weight >= 0.0, s"instance weight, $weight has to be >= 0.0")

    if (weight == 0.0) return this

    if (label == 1.0) {
      positiveClass = (positiveClass._1 + 1, positiveClass._2 + weight)
      this
    } else if (label == 0.0) {
      negativeClass = (negativeClass._1 + 1, negativeClass._2 + weight)
      this
    } else {
      totalInvalidCnt += 1
      this
    }
  }

  def merge(that: BinaryClassSummarizer): BinaryClassSummarizer = {
    totalInvalidCnt += that.totalInvalidCnt
    positiveClass = (positiveClass._1 + that.positiveClass._1, positiveClass._2 + that.positiveClass._2)
    negativeClass = (negativeClass._1 + that.negativeClass._1, negativeClass._2 + that.negativeClass._2)
    this
  }

  /** @return The total invalid input counts. */
  def countInvalid: Long = totalInvalidCnt

  def histogram: Array[(Long, Double)] = Array(negativeClass, positiveClass)

}
