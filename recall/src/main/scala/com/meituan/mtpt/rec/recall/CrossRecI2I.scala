package com.meituan.mtpt.rec.recall

import java.util.Calendar

import com.meituan.mtpt.rec.tools.{GeohashUtils, env, homePageRmdconfig}

import scala.collection.mutable.ArrayBuffer


/**
  * Created by yuanfei on 2018/07/10.
  */
object CrossRecI2I {
  val ALPHA = 0.3
  val SIM_ITEM_COUNT = 30
  val MIN_PRECISION = 0.000001

  def CrossRecSim(config: homePageRmdconfig, count: Int, countForAll: Int): Unit = {
    val stopTime = config.stopTime
    val startTime = config.startTime
    val stopTime_ = config.stopTime_
    val startTime_ = config.startTime_

    //到餐：226；酒店：209；旅游：217；休闲娱乐：3、5003；电影：page_bg 猫眼文化
    //丽人：2、5005；亲子：389、5010；结婚：388、5009；购物：379、5007；家装：600、5019；
    //教育培训：289、572；
    val user_page_view_sql =
      s"""
         |select if(item_id is null, "",item_id) item_id,
         |       if(uuid is null, "",uuid) uuid,
         |       if(item_type is null, "",item_type) item_type,
         |       if(event_timestamp is null, 0L,event_timestamp) event_timestamp,
         |       if(first_cate_id is null, "maoyan",first_cate_id) first_cate_id,
         |       if(second_cate_id is null, "film",second_cate_id) second_cate_id,
         |       if(page_city_id is null, "0",page_city_id) page_city_id,
         |       if(refer_page_name is null,"",refer_page_name) refer_page_name,
         |       if(latitude is null,"0",latitude) latitude,
         |       if(longitude is null,"0",longitude) longitude
         |  from mart_semantic.detail_platform_pageflow_daily
         | where partition_date >= '$startTime_'
         |   and uuid is not null
         |   and item_id is not null
         |   and item_id != ''
         |   and item_id > 0
         |   and item_type is not null
         |   and item_type in ('poi','deal','movie')
         |   and biz_bu != '外卖事业部'
         |   and page_bg != '外卖配送事业群'
         |   and event_timestamp is not null
         |   and page_stay_time > 1000
         |   and page_stay_time is not null
         |   and (first_cate_id in ("226","209","217","3","5003","2","5005","389","5010","388","5009","379","5007","600","5019","289","572") or page_bg = "猫眼文化")
         |   and page_city_id is not null
         |   and page_city_id > 0
         |   and event_timestamp is not null
         |   and event_timestamp > 0
         |   and latitude is not null
         |   and latitude > 0
         |   and longitude is not null
         |   and longitude > 0
       """.stripMargin


    val user_page_view = env.hsc.sql(user_page_view_sql).rdd.map(row => {
      val userid = row.getAs("uuid").toString
      val poiid = row.getAs("item_id").toString
      val cateid = row.getAs("first_cate_id").toString
      var cate = "daozong"
      val cate2 = row.getAs("second_cate_id").toString
      val latitude = row.getAs("latitude").toString.toDouble
      val longitude = row.getAs("longitude").toString.toDouble
      val geo5 = GeohashUtils.encodeLatLon(latitude, longitude, 5)
      val cityid = row.getAs("page_city_id").toString
      var itemtype = row.getAs("item_type").toString //poi、deal
      val timeStamp = row.getAs[Long]("event_timestamp")

      //判断是否是订单页
      val refer_page_name = row.getAs("refer_page_name").toString
      var isOrder = "0"
      if (("订单".r findFirstIn refer_page_name).size > 0) {
        isOrder = "1"
      }

      //判断业务
      if ("226".equals(cateid)) {
        cate = "daocan"
      } else if ("209".equals(cateid)) {
        cate = "hotel"
      } else if ("217".equals(cateid)) {
        cate = "lvyou"
      } else if ("3".equals(cateid) || "5003".equals(cateid)) {
        cate = "xiuyu"
      } else if ("2".equals(cateid) || "5005".equals(cateid)) {
        cate = "beauty"
      } else if ("389".equals(cateid) || "5010".equals(cateid)) {
        cate = "child"
      } else if ("388".equals(cateid) || "5009".equals(cateid)) {
        cate = "marry"
      } else if ("379".equals(cateid) || "5007".equals(cateid)) {
        cate = "shopping"
      } else if ("600".equals(cateid) || "5019".equals(cateid)) {
        cate = "decoration"
      } else if ("289".equals(cateid) || "572".equals(cateid)) {
        cate = "education"
      }
      else if ("maoyan".equals(cateid)) {
        if ("poi".equals(itemtype)) {
          cate = "film"
        } else if ("movie".equals(itemtype)) {
          itemtype = "deal"
          cate = "film"
        }
      }
      //(userid, Array((itemtype, poiid, cate, cityid, isOrder, timeStamp, cate2, geo5)))
      (userid, itemtype, poiid, cate, cityid, isOrder, timeStamp, cate2, geo5)
    }).filter(a => !a._4.equals("daozong") && a._2.length() > 1)
      .map {
        case (userid, itemtype, poiid, cate, cityid, isOrder, timeStamp, cate2, geo5) =>
          (userid, Array((itemtype, poiid, cate, cityid, isOrder, timeStamp, cate2, geo5)))
      }.reduceByKey(_ ++ _)
      .filter(r => r._2.size > 5 && r._2.size < 200)


    val result = user_page_view.flatMap {
      case (userid, poiList) =>
        val out = ArrayBuffer[((String, String, String), Int)]()
        for (poiInfo1 <- poiList) {
          val (itemtype1, poiid1, cate1, cityid1, isOrder1, timeStamp1, cate2_1, geo5_1) = poiInfo1
          var weight = 1
          if ("1".equals(isOrder1)) {
            weight += 1
          }
          for (poiInfo2 <- poiList) {
            val (itemtype2, poiid2, cate2, cityid2, isOrder2, timeStamp2, cate2_2, geo5_2) = poiInfo2
            if ("1".equals(isOrder2)) {
              weight += 8
            }
            if (cityid2.equals(cityid1)) {
              out.append(((itemtype1 + "_" + poiid1, itemtype2 + "_" + poiid2, cate2 + "_" + cityid2), 1))
            }
            if (timeStamp2 > timeStamp1 && cityid2.equals(cityid1)) {
              //添加poi维度的key
              if (!poiid2.equals(poiid1)) {
                out.append(((itemtype1 + "_" + poiid1, itemtype2 + "_" + poiid2, cate2 + "_" + cityid2), weight))
              }
            }
          }
        }
        out
    }.reduceByKey(_ + _).filter(_._2 > 3).map {
      case ((poiid, poiid2, cate), cnt) =>
        ((poiid, cate), Array((poiid2, cnt)))
    }.reduceByKey(_ ++ _)
      .filter(_._2.length > 3)
      .map {
        case ((poiid, cate), poiList) =>
          val sortedPoiList = poiList.sortWith((a, b) => a._2 > b._2).take(count)
          val maxCnt = sortedPoiList(1)._2.toDouble
          poiid + "_" + cate + "\t" + sortedPoiList.map { case (poiid2, cnt) =>
            val score = cnt * 1.0 / (maxCnt + 10.0)
            poiid2 + "_" + score.formatted("%.4f")
          }.mkString(";")
      }
    //poiid包含了城市item的类别： "poi_12345" 或者 "deal_12345"
    //cate包含了城市id(或者geohash5)，cate: cate + "_" + cityid

    val outputDir = s"/user/hadoop-recsys/yuanfei/CrossRecI2I"
    if (outputDir.contains("yuanfei") && env.hdfs.exists(new org.apache.hadoop.fs.Path(outputDir))) {
      env.hdfs.delete(new org.apache.hadoop.fs.Path(outputDir), true)
    }
    result.saveAsTextFile(outputDir)
  }

  def main(args: Array[String]) {
    val now_time = Calendar.getInstance().getTime
    val now = Calendar.getInstance()
    now.setTime(now_time)

    val stop = Calendar.getInstance()
    stop.setTimeInMillis(now.getTimeInMillis)
    stop.add(Calendar.DATE, -1)

    val time_gap = 2
    val start = Calendar.getInstance()
    start.setTimeInMillis(stop.getTimeInMillis)
    start.add(Calendar.DATE, -time_gap)

    val config = new homePageRmdconfig(now, stop, start, "")
    CrossRecSim(config, 50, 50)

    println("stop time is :" + config.stopTime_)
    println("start time is :" + config.startTime_)
    println("success~!")
  }
}
