package com.meituan.mtpt.rec.recallPractice

import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer
import com.meituan.mtpt.rec.tools.{ArgParser, HdfsUtils, env}
import org.apache.spark.sql.functions._
import env.hsc.implicits._

import scala.util.Random

object recallItem2Vec {
  var debug = true
  var vectorSize = 100
  var windowSize = 160
  var numPartitions = 1
  var maxIter = 1

  def genProductsSeq(priorData: RDD[(String, String, Int)], shuffleTimes: Int): DataFrame ={

    var seqData = priorData.map{
      case (orderId, productId, cartOrder) => (orderId, List((productId, cartOrder)))
    }.reduceByKey(_++_).map(r => r._2.sortBy(_._2).map(_._1))

    if(shuffleTimes > 0){
      seqData = seqData.flatMap(
        seq => {
          var newSeq = ArrayBuffer[List[String]](seq)
          for(_ <- 1 to shuffleTimes)  {
            newSeq += Random.shuffle(seq)
          }
          newSeq
        }
      )
    }

    if(debug){
      println("total training set size : " + seqData.count())
    }

    seqData.toDF("inp")
  }

  def trainWord2VecModel(seqData:DataFrame): Word2VecModel = {
    val word2vec = new Word2Vec()
      .setVectorSize(vectorSize)
      .setWindowSize(windowSize)
      .setMaxIter(maxIter)
      .setMinCount(0)
      .setNumPartitions(numPartitions)
      .setInputCol("inp")
      .setOutputCol("res")
    val model = word2vec.fit(seqData)
    model
  }

  def doRecall(testData:RDD[(String, String, String)],
               userHistory:RDD[(String, String, Int)],
               word2VecModel: Word2VecModel,
               orderBy:String="orderNum") : RDD[(String, List[(String, Float)])] = {

    var cmpFunc = (x:(String, Int, Int), y:(String, Int, Int)) => {
      (x._2 > y._2) || ((x._2 == y._2) && x._3 > y._3)  // 按orderNum排序
    }
    if(orderBy != "orderNum"){
      // 按orderTimes排序
      cmpFunc = (x:(String, Int, Int), y:(String, Int, Int)) => {
        (x._3 > y._3) || ((x._3 == y._3) && (x._2 > y._2))
      }
    }

    val userProductTimes = userHistory.map{
      case (userId, productId, _) => ((userId, productId), 1)
    }.reduceByKey(_+_)

    val userTriggers = userHistory.map{
      case (userId, productId, orderNum) => ((userId, productId), orderNum)    // userId, productId可能有多个orderNum，reduce取最大值
    }.reduceByKey((x, y) => math.max(x, y)).join(userProductTimes).map{
      case ((userId, productId), (orderNum, orderTimes)) => (userId, List((productId, orderNum, orderTimes)))
    }.reduceByKey(_++_).map(r => (r._1, r._2.sortWith(cmpFunc).take(20)))

    if(debug){
      println("userTriggers size : " + userTriggers.count().toString)
    }

    val myFindSynonyms = (trigger: String) => word2VecModel.findSynonymsArray(trigger, 10)

    val triggerList = userTriggers.flatMap(r => r._2.map(_._1)).distinct()
      .collect().toList

    var synonyms = ArrayBuffer[(String, Array[(String, Double)])]()
    triggerList.foreach(productId => {
      synonyms += ((productId, myFindSynonyms(productId)))
    })
    val triggerSynonyms = env.sc.parallelize(synonyms)

    if(debug){
      println("triggerSynonyms size : " + synonyms.size)
    }

    val productsRecall = testData.map(r => (r._2, r._1)).join(userTriggers).flatMap{
      case (userId, (orderId, seqTriggers)) =>
        seqTriggers.map(triggerTuple => (triggerTuple._1, orderId))
    }.join(triggerSynonyms).map{
      case (trigger, (orderId, seq)) =>
        (orderId, seq.map(x => (x._1, x._2.toFloat)).toList)
    }.reduceByKey(_++_)

    productsRecall
  }

  def metric(recallItems:RDD[(String, List[(String, Float)])], testData:RDD[(String, String, String)]) : Unit = {
    val recallHits = testData.map(r => (r._1, List(r._3))).reduceByKey(_++_).join(recallItems).map{
      case (_, (seqGT, seqRecall)) =>
        val seqRecallDistinct = seqRecall.map(_._1).distinct   // 这里要给recall去重！
        val P = seqGT.size.toFloat
        val TP = seqGT.intersect(seqRecallDistinct).size.toFloat
        val recallSize = seqRecallDistinct.size
        (TP, P, recallSize, 1)
    }.reduce((x, y) => (x._1 + y._1, x._2 + y._2, x._3 + y._3, x._4+y._4))

    val rate = recallHits._1 / recallHits._2
    println(recallHits._1.toString + " " + recallHits._2.toString + " " + rate.toString)
    if(debug){
      println("total recalled items: " + recallHits._3.toString)
    }
  }

  def main(args: Array[String]): Unit = {
    val argMap = ArgParser.parseAsMap(args)
    println(argMap)
    vectorSize = argMap.getOrElse("vectorSize", "100").toInt
    windowSize = argMap.getOrElse("windowSize", "160").toInt
    debug = argMap.getOrElse("debug", "false").toBoolean
    val shuffleTimes = argMap.getOrElse("shuffleTimes", "0").toInt
    numPartitions = argMap.getOrElse("numPartitions", "128").toInt
    val orderBy = argMap.getOrElse("orderBy", "orderNum")
    maxIter = argMap.getOrElse("maxIter", "1").toInt
    println((vectorSize, windowSize, shuffleTimes, maxIter, orderBy))

    val data = loadData()
    val priorData = data._1
    val testData = data._2
    val userHistory = data._3

    val seqData = genProductsSeq(priorData, shuffleTimes)
    val model = trainWord2VecModel(seqData)

    val seqRecall = doRecall(testData, userHistory, model, orderBy)

    if(debug){
      println("total recall size : " + seqRecall.count())
    }

    val baseDir = "/user/hadoop-recsys/jiangzelong02/recstar/item2Vec/"
    val path = baseDir + s"v${vectorSize}_w${windowSize}_${orderBy}_s${shuffleTimes}"
    HdfsUtils.saveRecallList(seqRecall, path)

    metric(seqRecall, testData)
  }


  def loadData(): (RDD[(String, String, Int)], RDD[(String, String, String)], RDD[(String, String, Int)]) ={
    val priorDataSql =
      s"""
       |select t_prior.order_id order_id,
       |       t_prior.product_id product_id,
       |       add_to_cart_order
       |  from ba_dealrank.recommend_star_order_products__prior t_prior
    """.stripMargin

    val testDataSql =
      s"""
         |  select t_train.order_id order_id,
         |       t_train.product_id product_id,
         |       user_id
         |  from ba_dealrank.recommend_star_order_products__train t_train
         |  left join ba_dealrank.recommend_star_orders t_order
         |    on t_order.order_id = t_train.order_id
      """.stripMargin

    val userHistorySql =
      s"""
         |select t_prior.order_id order_id,
         |       t_prior.product_id product_id,
         |       user_id,
         |       order_number
         |  from ba_dealrank.recommend_star_order_products__prior t_prior
         |  left join ba_dealrank.recommend_star_orders t_order
         |    on t_order.order_id = t_prior.order_id
  """.stripMargin

    val priorData = env.hsc.sql(priorDataSql).rdd.map(
      row => {
        val orderId = row.getAs("order_id").toString
        val productId = row.getAs("product_id").toString
        val cartOrder = row.getAs[Int]("add_to_cart_order")
        (orderId, productId, cartOrder)
      }
    )

    val testData = env.hsc.sql(testDataSql).rdd.map(
      row => {
        val orderId = row.getAs("order_id").toString
        val userId = row.getAs("user_id").toString
        val productId = row.getAs("product_id").toString
        (orderId, userId, productId)
      }
    )

    val userHistory = env.hsc.sql(userHistorySql).rdd.map(
      row => {
        val productId = row.getAs("product_id").toString
        val userId = row.getAs("user_id").toString
        val orderNum = row.getAs[Int]("order_number")
        (userId, productId, orderNum)
      }
    )

    (priorData, testData, userHistory)
  }
}
