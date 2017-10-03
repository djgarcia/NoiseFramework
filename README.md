# NoiseFramework

This framework implements two Big Data preprocessing approaches to remove noisy examples: an homogeneous ensemble (HME-BD) and an heterogeneous ensemble (HTE-BD) filter, with special emphasis in their scalability and performance traits. A simple filtering approach based on similarities between instances (ENN-BD) is also implemented.

This software has been proved with four large real-world datasets such as:
- SUSY dataset: 5M instances and 18 attributes. https://archive.ics.uci.edu/ml/datasets/SUSY
- HIGGS dataset: 11M instances and 28 attributes. https://archive.ics.uci.edu/ml/datasets/HIGGS
- Epsilon dataset: 400K instances and 2K attributes. http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#epsilon
- A dataset selected for the GECCO-2014 in Vancouver, July 13th, 2014 competition, which comes from the Protein Structure Prediction field (http://cruncher.ncl.ac.uk/bdcomp/).
We have created a a reduced version of this dataset with 1M instances, 631 attributes and 2 classes.

## Brief benchmark results:

* HME-BD has shown to be the best noise filter algorithm, achieving the best accuracy.
* HME-BD is also the most efficient in terms of computing time.
* HTE-BD can outperform HME-BD for some datasets under low levels of noise.

## Example (HME-BD)


```scala
import org.apache.spark.mllib.feature._

val nTrees = 100
val maxDepth = 10
val k = 4

// Data must be cached in order to improve the performance

val hme_bd_model = new HME_BD(trainingData, // RDD[LabeledPoint]
                              nTrees, // size of the Random Forests
                              k, // number of partitions
                              maxDepth, // depth of the Random Forests
                              seed) // seed for the Random Forests

val hme_bd = hme_bd_model.runFilter()
```

## Example (HTE-BD)


```scala
import org.apache.spark.mllib.feature._

val nTrees = 100
val maxDepth = 10
val k = 4
val voting = 0 // 0 = majority, 1 = consensus

// Data must be cached in order to improve the performance

val hte_bd_model = new HTE_BD(trainingData, // RDD[LabeledPoint]
                              nTrees, // size of the Random Forests
                              k, // number of partitions
                              vote, // voting strategy
                              numClass, // number of classes
                              numFeatures, // number of features
                              maxDepth, // depth of the Random Forests
                              seed) // seed for the Random Forests

val hte_bd = hte_bd_model.runFilter()
```

## Example (ENN-BD)


```scala
import org.apache.spark.mllib.feature._

// Data must be cached in order to improve the performance

val enn_bd_model = new ENN_BD(trainingData, // RDD[LabeledPoint]
                              k) // number of neighbors

val enn_bd = enn_bd_model.runFilter()
```
