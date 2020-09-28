package com.meituan.mtpt.rec.recallPractice

import com.meituan.mtpt.rec.tools.{env, HdfsUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

object recallHot {

  def main(args: Array[String]): Unit = {

    val priorData = env.hsc.sql("""
                                  |select t_prior.order_id order_id,
                                  |       t_prior.product_id product_id,
                                  |       reordered,
                                  |       order_dow,
                                  |       order_hour_of_day
                                  |  from ba_dealrank.recommend_star_order_products__prior t_prior
                                  |  left join ba_dealrank.recommend_star_orders t_order
                                  |    on t_order.order_id = t_prior.order_id
  """.stripMargin
    ).rdd

    val testData = env.hsc.sql("""
                                 |select t_train.order_id order_id,
                                 |       t_train.product_id product_id,
                                 |       reordered,
                                 |       order_dow,
                                 |       order_hour_of_day
                                 |  from ba_dealrank.recommend_star_order_products__train t_train
                                 |  left join ba_dealrank.recommend_star_orders t_order
                                 |    on t_order.order_id = t_train.order_id
  """.stripMargin
    ).rdd

    // 按小时x天级获取热单列表
    val hotProductsByHourDay = priorData.map{
      row =>
        val productId = row.getAs("product_id").toString
        val dow = row.getAs("order_dow").toString
        val hod = row.getAs("order_hour_of_day").toString
        ((productId, dow, hod), 1)
    }.reduceByKey(_+_).map{
      case ((productId, dow, hod), cnt) => ((dow, hod), List((productId, cnt.toFloat)))
    }.reduceByKey(_ ++ _).map{
      case ((dow, hod), seq) => ((dow, hod), seq.sortBy(-_._2).take(20))
    }

    val resultByHourDay = testData.map{
      row =>
        val orderId = row.getAs("order_id").toString
        val dow = row.getAs("order_dow").toString
        val hod = row.getAs("order_hour_of_day").toString
        val productId = row.getAs("product_id").toString
        ((orderId, dow, hod), List(productId))
    }.reduceByKey(_ ++ _).map{
      case ((order_id, dow, hod), seq) => ((dow, hod), (order_id, seq))
    }.join(hotProductsByHourDay)


    val metric = resultByHourDay.map{
      case ((dow, hod), ((order_id, seqGT), seqHot)) =>
        val setGT = seqGT.toSet
        val P = seqGT.size.toFloat
        var TP = 0f
        seqHot.foreach{
          case (product, cnt) =>
            if(setGT.contains(product)){
              TP += 1
            }
        }
        (TP, P)
    }.reduce((x,y) => (x._1+y._1, x._2+y._2))

    println(metric._1 / metric._2)

    val recallList = resultByHourDay.map{
      case (_, ((orderId, _), seqHot)) => (orderId, seqHot)
    }

    HdfsUtils.saveRecallList(recallList, s"/user/hadoop-recsys/jiangzelong02/recstar/recallHot")
  }
}
