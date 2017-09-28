package org.apache.spark.mllib.feature

import org.apache.spark.mllib.classification.kNN_IS.kNN_IS
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

class ENN_BD(val data: RDD[LabeledPoint], val k: Int = 1) extends Serializable {

  def runFilter(): RDD[LabeledPoint] = {

    val numClass = data.map(_.label).distinct().collect().length
    val numFeatures = data.first().features.size

    val knn = kNN_IS.setup(data, data, k, 2, numClass, numFeatures, data.partitions.length, 64, -1, 1)

    val predictions = knn.predict(data.context).map(_._1).zipWithIndex.map { case (key, v) => (v, key) }

    val filteredData = data.zipWithIndex.map { case (key, v) => (v, key) }.join(predictions).map {
      case (key, (l, pred)) =>
        if (l.label == pred) {
          l
        } else {
          LabeledPoint(-1, l.features)
        }
    }.filter { point => point.label != -1 }

    return filteredData
  }
}
