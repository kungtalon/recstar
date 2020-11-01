package com.meituan.mtpt.rec.tools

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import shapeless.syntax.std.tuple.productTupleOps

/**
  * Created by feng on 2017/11/2.
  */
object HdfsUtils {

  def hdfsDelete(path: String): Unit = {
    val hdfsPath = new Path(path)
    val hconf = new Configuration()
    val hdfs = FileSystem.get(hconf)

    if (hdfs.exists(hdfsPath)) {
      try {
        hdfs.delete(hdfsPath, true)
      } catch {
        case _: Throwable => {}
      }
    }
  }

  def exists(path: String): Boolean = {
    val hdfsPath = new Path(path)
    val hconf = new Configuration()
    val hdfs = FileSystem.get(hconf)
    hdfs.exists(hdfsPath)
  }

  def saveRecallList(recallRDD : RDD[(String, List[(String, Float)])], path:String) : Unit ={
    if(exists(path)){
      hdfsDelete(path)
    }

    println("saving recall results to " + path)
    recallRDD.map{
      case (orderId, seq) =>
        var rowStr = orderId + ":"
        val set = seq.toSet
        for(pair <- set){
          rowStr += pair._1 + "_" + pair._2.toString + ";"
        }
        rowStr
    }.saveAsTextFile(path)

  }

  def readRecallList(path:String) : RDD[(String, List[(String, Float)])] ={
    if(!exists(path)){
      println("wrong path!")
      return null
    }

    println("reading recall results from " + path)
    val recallRDD = env.sc.textFile(path).map{
      rowStr:String =>
        val splitted = rowStr.split(":")
        val orderId = splitted(0)
        val products = splitted(1).split(";").map(_.split("_")).map(
          pair => (pair(0), pair(1).toFloat)
        ).toList
        (orderId, products)
    }

    recallRDD
  }

  def saveOrderLists(resultRDD:RDD[(String, Int, Int, Float, List[Int], List[Int], List[Int], List[Int])], path:String): Unit = {
    resultRDD.map{
      case (orderId, dow, hod, daysPo, histOrdered, histReordered, ordered, subsampled) =>
        val histOrderedStr = histOrdered.mkString(" ")
        val histReorderedStr = histReordered.mkString(" ")
        val orderedStr = ordered.mkString(" ")
        val subsampledStr = subsampled.mkString(" ")
        List(orderId, dow, hod, daysPo, histOrderedStr, histReorderedStr, orderedStr, subsampledStr).mkString(",")
    }.saveAsTextFile(path)
  }
}
