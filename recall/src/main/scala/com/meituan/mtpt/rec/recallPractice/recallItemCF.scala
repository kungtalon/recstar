package com.meituan.mtpt.rec.recallPractice

import com.meituan.mtpt.rec.tools.{env, HdfsUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

object recallItemCF {
  var debug: Boolean = false


  // 相似度计算函数：
  def calSim(productsCnt:RDD[(String, Float)], concurCnt:RDD[((String, String), Float, Float)]) : RDD[(String, List[(String, Float)])] = {
    val productsScore = productsCnt.map{
      case(product, score)=>(product, score*score)
    }.reduceByKey(_+_).map(r => (r._1, math.sqrt(r._2).toFloat))

    if(debug){
      println("productsScore size : " + productsScore.count().toString)
    }

    val concurScore = concurCnt.map{
      case (k, v1, v2) => (k, v1*v2)
    }.reduceByKey(_+_).map{
      case ((p1, p2), v) => (p1, (p2, v))
    }.join(productsScore).map{
      case (p1, ((p2, v), s1)) => (p2, (p1, v, s1))
    }.join(productsScore).map{
      case (p2, ((p1, v, s1), s2)) => (p1, List((p2, v/(s1*s2))))
    }.reduceByKey(_++_).map{
      case (p1, seq) => (p1, seq.sortBy(-_._2).take(10))
    }

    concurScore
  }

  def getProductsSim(priorDataRDD:RDD[Row],
                     priorDataConcurRDD:RDD[Row],
                     reorderWeight:Int) : RDD[(String, List[(String, Float)])] = {
    val priorProductsCnt = priorDataRDD.map{
      row =>
        val orderId = row.getAs("order_id").toString
        val productId = row.getAs("product_id").toString
        val reordered = row.getAs[Integer]("reordered")
        (productId, 1f + reorderWeight * reordered)
    }

    val priorConcurCnt = priorDataConcurRDD.map{
      row =>
        val productId1 = row.getAs("product_id1").toString
        val productId2 = row.getAs("product_id2").toString
        val reordered1 = row.getAs[Integer]("reordered1")
        val reordered2 = row.getAs[Integer]("reordered2")
        ((productId1, productId2), 1f + reorderWeight * reordered1, 1f + reorderWeight * reordered2)
    }

    if(debug){
      println("priorProductsCnt size : " + priorProductsCnt.count().toString)
      println("priorConcurCnt size : " + priorProductsCnt.count().toString)
    }
    calSim(priorProductsCnt, priorConcurCnt)
  }

  def doRecall(testData:RDD[(String, String, String)], userHistory:RDD[(String, (String, Int))],
               lookUpTable:RDD[(String, List[(String, Float)])], orderBy:String="orderNum") : RDD[(String, List[(String, Float)])] = {
    // testData: (orderId, userId, productId)
    // userHistory: (userId, (productId, orderNum))
    // lookUpTable: (productId, seqProducts)
    // 按照(orderNum, orderTimes)或(orderTimes, orderNum)进行排序

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
      case (userId, (productId, orderNum)) => ((userId, productId), 1)
    }.reduceByKey(_+_)

    val userTriggers = userHistory.map{
      case (userId, (productId, orderNum)) => ((userId, productId), orderNum)    // userId, productId可能有多个orderNum，reduce取最大值
    }.reduceByKey((x, y) => math.max(x, y)).join(userProductTimes).map{
      case ((userId, productId), (orderNum, orderTimes)) => (userId, List((productId, orderNum, orderTimes)))
    }.reduceByKey(_++_).map(r => (r._1, r._2.sortWith(cmpFunc).take(20)))

    if(debug){
      println("userTriggers size : " + userTriggers.count().toString)
    }

    val productsRecall = testData.map(r => (r._2, r._1)).join(userTriggers).flatMap{
      case (userId, (orderId, seqTriggers)) =>
        seqTriggers.map(triggerTuple => (triggerTuple._1, (userId, orderId)))
    }.join(lookUpTable).map{
      case (trigger, ((userId, orderId), seqRecall)) => (orderId, seqRecall)
    }.reduceByKey(_++_)

    productsRecall
  }

  def metric(recallRDD:RDD[(String, List[(String, Float)])], testData:RDD[(String, String, String)]) : Unit = {
    val recallHits = testData.map(r => (r._1, List(r._3))).reduceByKey(_++_).join(recallRDD).map{
      case (_, (seqGT, seqRecall)) =>
        val setRecall = seqRecall.map(_._1).toSet   // 这里要给recall去重！
        val P = seqGT.size.toFloat
        val TP = seqGT.count(product => setRecall.contains(product)).toFloat
        (TP, P)
    }.reduce((x, y) => (x._1 + y._1, x._2 + y._2))

//    println("debug 2" + recallHits2.toString)
    val rate = recallHits._1 / recallHits._2
    println(recallHits._1.toString + " " + recallHits._2.toString + " " + rate.toString)
  }

  def main(args: Array[String]): Unit ={
    val reorderWeight = args(0).toInt
    var orderBy = "orderNum"
    if(args.length >= 2){
      orderBy = args(1)
      if(args.length >= 3){
        debug = args(2).toBoolean
      }
    }

    val priorData = env.hsc.sql(getSQL("priorData")).rdd

    val priorDataConcur = env.hsc.sql(getSQL("priorDataConcur")).rdd

    val testDataRDD = env.hsc.sql(getSQL("testDataRDD")).rdd.map(
      row => {
        val orderId = row.getAs("order_id").toString
        val productId = row.getAs("product_id").toString
        val user_id = row.getAs("user_id").toString
        (orderId, user_id, productId)
      }
    )

    val userHistory = env.hsc.sql(getSQL("userHistory")).rdd.map(
      row => {
        val productId = row.getAs("product_id").toString
        val userId = row.getAs("user_id").toString
        val orderNum = row.getAs[Int]("order_number")
        (userId, (productId, orderNum))
      }
    )

    val concurRDD = getProductsSim(priorData, priorDataConcur, reorderWeight)

    val productsRecall = doRecall(testDataRDD, userHistory, concurRDD, orderBy)

    if(debug){
      println("lookup table size : " + concurRDD.count().toString)
      println("recall result size : " + productsRecall.count().toString)
      println(testDataRDD.map(r => (r._1, List(r._3))).reduceByKey(_++_).count())
      return
    }

    HdfsUtils.saveRecallList(productsRecall, s"/user/hadoop-recsys/jiangzelong02/recstar/itemCF_${reorderWeight}_$orderBy")

    println("evaluating")
    metric(productsRecall, testDataRDD)

  }

  def getSQL(dataName: String): String ={
    val priorData = """
  |select t_prior.order_id order_id,
  |       t_prior.product_id product_id,
  |       reordered
  |  from ba_dealrank.recommend_star_order_products__prior t_prior
    """.stripMargin

    val priorDataConcur = s"""
  |select t_prior1.order_id order_id,
  |       t_prior1.product_id product_id1,
  |       t_prior1.reordered reordered1,
  |       t_prior2.product_id product_id2,
  |       t_prior2.reordered reordered2
  |  from ba_dealrank.recommend_star_order_products__prior t_prior1
  |  left join ba_dealrank.recommend_star_order_products__prior t_prior2
  |    on t_prior1.order_id = t_prior2.order_id
    """.stripMargin

    val testDataRDD =
      s"""
  |  select t_train.order_id order_id,
  |       t_train.product_id product_id,
  |       user_id
  |  from ba_dealrank.recommend_star_order_products__train t_train
  |  left join ba_dealrank.recommend_star_orders t_order
  |    on t_order.order_id = t_train.order_id
      """.stripMargin

    val userHistory =
      s"""
  |select t_prior.order_id order_id,
  |       t_prior.product_id product_id,
  |       user_id,
  |       order_number
  |  from ba_dealrank.recommend_star_order_products__prior t_prior
  |  left join ba_dealrank.recommend_star_orders t_order
  |    on t_order.order_id = t_prior.order_id
  """.stripMargin

    val res = dataName match {
      case "priorData" => priorData
      case "priorDataConcur" => priorDataConcur
      case "testDataRDD" => testDataRDD
      case "userHistory" => userHistory
    }
    res
  }
}
