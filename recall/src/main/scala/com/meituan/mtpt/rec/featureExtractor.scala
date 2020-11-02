package com.meituan.mtpt.rec

import com.meituan.mtpt.rec.consts
import com.meituan.mtpt.rec.tools.{env, HdfsUtils}
import org.apache.spark.rdd.RDD
import scala.util.Random

/**
 * 抽取hist_order_list, hist_reorder_list, this_order_list, subsamples
 */
object featureExtractor {
  val hist_order_limit = 75
  val hist_reorder_limit = 20
  val negative_sample_size = 120
  val item_count = 49688
  val seed = 0

  def getRawTable(): RDD[(String, String, Int, Int, Int, Int)] ={
    val sqlBar = consts.allOrders()
    val res = env.hsc.sql(sqlBar).rdd.map(
      row => {
        val orderId = row.getAs("order_id").toString
        val userId = row.getAs("user_id").toString
        val productId = row.getAs[Int]("product_id")
        val orderNum = row.getAs[Int]("order_number")
        val cartOrder = row.getAs[Int]("add_to_cart_order")
        val reordered = row.getAs[Int]("reordered")
        (orderId, userId, productId, orderNum, cartOrder, reordered)
      }
    )
    res
  }

  def getOrdersContext(evalSet: String): RDD[(String, (Int, Int, Float))] ={
    val sqlBar = consts.mainOrdersContext(evalSet)
    val res = env.hsc.sql(sqlBar).rdd.map(
      row => {
        val orderId = row.getAs("order_id").toString
        val order_dow = row.getAs[Int]("order_dow")
        val order_hod = row.getAs[Int]("order_hour_of_day")
        val daysPo = row.getAs[Float]("days_since_prior_order")
        (orderId, (order_dow, order_hod, daysPo))
      }
    )
    res
  }

  def mergeUserHist(originRDD:
                     RDD[(String, String, Int, Int, Int, Int)]): RDD[(String, List[(Int, Int,Int,Int)])] = {
    val res = originRDD.map{
      case (_, userId, productId, orderNum, cartOrder, reordered) =>
        (userId, List((productId, orderNum, orderNum*200 + cartOrder, reordered)))
    }.reduceByKey(_++_).map(row => (row._1, row._2.sortBy(_._3)))
    res
  }

  def getHistOrderList(allMerged: RDD[(String, ((String, Int), List[(Int, Int, Int, Int)]))]): RDD[(String, List[Int])] = {
    val orderHist = allMerged.map{
      case (_, (orderInfo, histSeq)) =>
        val cutHistList = histSeq.filter(_._2 < orderInfo._2)
          .sortBy(-_._3).take(hist_order_limit).map(_._1)
        (orderInfo._1, cutHistList)
    }
    orderHist
  }

  def getHistReorderList(allMerged: RDD[(String, ((String, Int), List[(Int, Int, Int, Int)]))]): RDD[(String, List[Int])] = {
    val reorderHist = allMerged.map{
      case (_, (orderInfo, histSeq)) =>
        val cutHistList = histSeq.filter(t => (t._2 < orderInfo._2) && (t._4 == 1))
          .sortBy(-_._3).take(hist_reorder_limit).map(_._1)
        (orderInfo._1, cutHistList)
    }
    reorderHist
  }

  def getThisOrderList(allMerged: RDD[(String, ((String, Int), List[(Int, Int, Int, Int)]))]): RDD[(String, List[Int])] ={
    val thisOrderList = allMerged.map{
      case (_, (orderInfo, histSeq)) =>
        val orderList = histSeq.filter(_._2 == orderInfo._2).map(_._1)
        (orderInfo._1, orderList)
    }
    thisOrderList
  }

  def getNegativeSamples(thisOrderList: RDD[(String, List[Int])]):RDD[(String, List[Int])] = {
    val rng = new Random(seed)
    val allList = (1 to item_count).toList
    val subsampled = thisOrderList.flatMap{
      case (orderId, orderList) =>
        val sampleList = allList.diff(orderList)
        for (i <- sampleList) yield {
          val sampled = i +: rng.shuffle(sampleList).take(negative_sample_size)
          (orderId, sampled)
        }
    }
    subsampled
  }

  def main(args: Array[String]): Unit ={
    val originRDD = getRawTable()
    val userHistory = mergeUserHist(originRDD)

    val orders = originRDD.map{
      case (orderId, userId, _, orderNum, _, _) =>
        (userId, (orderId, orderNum))
    }.distinct().filter(r => r._2._2 > 1)
    println("orders count: " + orders.count())

    val allMerged = orders.join(userHistory)

    val histOrderList = getHistOrderList(allMerged)
    val histReorderList = getHistReorderList(allMerged)
    val thisOrderList = getThisOrderList(allMerged)
    val subsampledRDD = getNegativeSamples(thisOrderList)

    val joinRDD = histOrderList.leftOuterJoin(histReorderList).map{
      case (orderId, (histOrdered, Some(histReordered))) =>
        (orderId, (histOrdered, histReordered))
      case (orderId, (histOrdered, None)) =>
        (orderId, (histOrdered, List()))
    }.join(thisOrderList).join(subsampledRDD).map{
      case (orderId, (((histOrdered, histReordered), ordered), subsampled)) =>
        (orderId, (histOrdered, histReordered, ordered, subsampled))
    }

    val priorOrders = getOrdersContext("prior")
    val trainOrders = getOrdersContext("train")
    val priorRDD = priorOrders.join(joinRDD).map{
      case (orderId, (ctx, hist)) =>
        (orderId, ctx._1, ctx._2, ctx._3, hist._1, hist._2, hist._3, hist._4)
    }
    val trainRDD = trainOrders.join(joinRDD).map{
      case (orderId, (ctx, hist)) =>
        (orderId, ctx._1, ctx._2, ctx._3, hist._1, hist._2, hist._3, hist._4)
    }

    println("joined count: " + joinRDD.count().toString)
    HdfsUtils.saveOrderLists(priorRDD, s"/user/hadoop-recsys/jiangzelong02/recstar/priorInput")
    HdfsUtils.saveOrderLists(trainRDD, s"/user/hadoop-recsys/jiangzelong02/recstar/trainInput")
    println("Done!")
  }

}
