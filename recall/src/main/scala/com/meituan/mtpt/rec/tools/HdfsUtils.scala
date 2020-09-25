package com.meituan.mtpt.rec.tools

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

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

}
