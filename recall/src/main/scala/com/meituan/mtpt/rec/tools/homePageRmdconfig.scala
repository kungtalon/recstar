package com.meituan.mtpt.rec.tools

import java.text.SimpleDateFormat
import java.util.Calendar

class homePageRmdconfig(now:Calendar, stop:Calendar, start:Calendar, val biz_path:String="") {


  val sdf = new SimpleDateFormat("yyyyMMdd")
  val sdf_ = new SimpleDateFormat("yyyy-MM-dd")

  val startTime = sdf.format(start.getTime)
  val startTime_ = sdf_.format(start.getTime)
  val startTimeUnix = start.getTimeInMillis/1000L

  val stopTime = sdf.format(stop.getTime)
  val stopTime_ = sdf_.format(stop.getTime)
  val stopTimeUnix = stop.getTimeInMillis/1000L + 3600*24

  val nowTime = sdf.format(now.getTime)
  val nowTime_ = sdf_.format(now.getTime)
  val nowTimeUnix = now.getTimeInMillis/1000L + 3600*24

}

class homePageRmdconfig_new(now_new:Calendar, now:Calendar, stop:Calendar, start:Calendar, val biz_path:String="") {


  val sdf = new SimpleDateFormat("yyyyMMdd")
  val sdf_ = new SimpleDateFormat("yyyy-MM-dd")

  val startTime = sdf.format(start.getTime)
  val startTime_ = sdf_.format(start.getTime)
  val startTimeUnix = start.getTimeInMillis/1000L

  val stopTime = sdf.format(stop.getTime)
  val stopTime_ = sdf_.format(stop.getTime)
  val stopTimeUnix = stop.getTimeInMillis/1000L + 3600*24

  val nowTime = sdf.format(now.getTime)
  val nowTime_ = sdf_.format(now.getTime)
  val nowTimeUnix = now.getTimeInMillis/1000L + 3600*24


  val now_newTime = sdf.format(now_new.getTime)
  val now_newTime_ = sdf_.format(now_new.getTime)
  val now_newTimeUnix = now_new.getTimeInMillis/1000L + 3600*24
}

