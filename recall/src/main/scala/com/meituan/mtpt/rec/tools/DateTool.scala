package com.meituan.mtpt.rec.tools

import java.text.SimpleDateFormat
import java.util.Calendar

object DateTool {
  //根据当前日期获取昨天日期
  def yesterday(): String ={
    val  dateFormat:SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd")
    val cal:Calendar=Calendar.getInstance()
    cal.add(Calendar.DATE,-1)
    val yesterday=dateFormat.format(cal.getTime())
    yesterday
  }

  //指定日期及间隔天数，获取n天前日期
  def getDaysBefore(dt: String, interval: Int): String = {
    val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd")

    val cal: Calendar = Calendar.getInstance()
    cal.setTime(dateFormat.parse(dt));

    cal.add(Calendar.DATE, - interval)
    val yesterday = dateFormat.format(cal.getTime())
    yesterday
  }

  //指定日期及间隔天数，获取n天前日期
  def getDaysBeforeKey(dt: String, interval: Int): String = {
    val dateFormat: SimpleDateFormat = new SimpleDateFormat("yyyy-MM-dd")

    val cal: Calendar = Calendar.getInstance()
    cal.setTime(dateFormat.parse(dt));

    cal.add(Calendar.DATE, - interval)

    val dateFormatKey: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
    val yesterday = dateFormatKey.format(cal.getTime())
    yesterday
  }
}