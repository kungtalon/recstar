package com.meituan.mtpt.rec.tools

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD

/**
  * Created by jiangzelong on 2020/09/28
  */
object ArgParser {

  def parseAsMap(args: Array[String]): Map[String, String] ={
    var argMap = Map[String, String]()
    for(row <- args){
      val sp = row.split("=")
      argMap += (sp(0) -> sp(1))
    }
    argMap
  }

}
