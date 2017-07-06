package org.apache.spark.mllib.feature

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.kNN_IS.kNN_IS
import org.apache.spark.mllib.util.MLUtils

class HTE_BD(val data: RDD[LabeledPoint], val nTrees: Int, val k: Int, val vote: Int, val numClass: Int, val numFeatures: Int, val maxDepth: Int = 10, val seed: Int) extends Serializable {

  private val labels = data.map(_.label).distinct().collect()
  private var modelNoise: Array[RDD[LabeledPoint]] = new Array[RDD[LabeledPoint]](k)

  def runFilter(): RDD[LabeledPoint] = {

    val cvdat = MLUtils.kFold(data, k, seed)

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

          val lrModel = new LogisticRegressionWithLBFGS()
            .setNumClasses(labels.length)
            .run(train)

          val knnModel = kNN_IS.setup(train, test, 1, 2, numClass, numFeatures, data.partitions.size, 64, -1, 1)

          val rfModel = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo,
            nTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed)

          val knnPred = knnModel.predict(train.context).map(_._1).collect()
          val rfPred = rfModel.predict(test.map(_.features)).collect()
          val lrPred = lrModel.predict(test.map(_.features)).collect()

          val joinedPreds = test.zipWithIndex.map {
            case (example, v) =>
              var noiseCount = 0
              val i = v.toInt

              if (rfPred(i) != example.label) noiseCount += 1
              if (lrPred(i) != example.label) noiseCount += 1
              if (knnPred(i) != example.label) noiseCount += 1

              if (vote == 0) {
                if (noiseCount >= 2) {
                  LabeledPoint(-1, example.features)
                } else {
                  example
                }
              } else {
                if (noiseCount == 3) {
                  LabeledPoint(-1, example.features)
                } else {
                  example
                }
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
