package com.meituan.mtpt.rec.tools

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{SparkConf, SparkContext}

object env {

  val conf = new SparkConf()
  val sc = new SparkContext(conf)
  val hsc = new HiveContext(sc)

  val hdfs_conf=new Configuration()
  val hdfs=FileSystem.get(hdfs_conf)


}

