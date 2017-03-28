package org.apache.spark.mllib.feature

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.mllib.util.MLUtils

//Train with K-1, predict with 1

class HME_BD(val data: RDD[LabeledPoint], val nTrees: Int, val k: Int, val maxDepth: Int = 10, val seed: Int) extends Serializable {

  private val labels = data.map(_.label).distinct().collect()
  private var modelNoise: Array[RDD[LabeledPoint]] = new Array[RDD[LabeledPoint]](k)

  def runFilter(): RDD[LabeledPoint] = {

    val cvdat = MLUtils.kFold(data, k, 12345)

    //RF Parameters
    val numClasses = labels.length
    val categoricalFeaturesInfo = Map[Int, Int]()
    val featureSubsetStrategy = "all"
    val impurity = "gini"
    val maxBins = 32

    modelNoise = cvdat.map {
      case (train, test) =>
        {

          var filteredData: RDD[LabeledPoint] = data.context.emptyRDD

          val rfModel = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo,
            nTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed)

          val rfPred = rfModel.predict(test.map(_.features)).zipWithIndex.map { case (k, v) => (v, k) }

          val joinedPreds = test.zipWithIndex.map { case (k, v) => (v, k) }.join(rfPred).map { l =>
            val example = l._2._1
            val rfVote = l._2._2

            if (rfVote != example.label) {
              LabeledPoint(-1, example.features)
            } else {
              example
            }

          }.filter { point => point.label != -1 }
          filteredData.union(joinedPreds)
        }
    }

    var filteredData = modelNoise(0).filter { point => point.label != -1 }

    for (i <- 1 to k - 1) {
      filteredData = filteredData.union(modelNoise(i).filter { point => point.label != -1 })
    }

    return filteredData
  }
}
