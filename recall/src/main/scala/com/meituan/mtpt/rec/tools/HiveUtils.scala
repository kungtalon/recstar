package com.meituan.mtpt.rec.tools

import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by gaozhen on 2019/2/14.
  */
object HiveUtils {

  def writeHiveTable(sqlContext: SparkSession, tempDF: DataFrame,
                     outputTable: String, partition: String, hiveConf: String
                    ): Unit = {
    val schema = tempDF.schema
    val colNames = schema.fields.map(_.name).mkString(",")
    tempDF.createOrReplaceTempView("tmp_table")
    sqlContext.sql("SET hive.exec.dynamic.partition=true")
    sqlContext.sql("SET hive.exec.dynamic.partition.mode=nostrick")
    if (hiveConf != null) {
      sqlContext.sql(hiveConf)
    }
    val field = schema.fields.map {
      schema => schema.name + " " + (if (schema.dataType.typeName.equals("integer")) "int" else schema.dataType.typeName)
    }.mkString(",")

    println("fields:", field)

    sqlContext.sql(s"create table if not exists $outputTable($field) " +
      s"partitioned by (dt string) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' LINES TERMINATED BY '\n'"
    )

    //sqlContext.sql(s"alter table $outputTable drop if exists partition (dt=$partition)")

    println("sqlText:", s"insert overwrite table $outputTable partition(dt=$partition) SELECT $colNames FROM tmp_table")
    sqlContext.sql(s"insert overwrite table $outputTable partition(dt=$partition) SELECT $colNames FROM tmp_table")

  }

  def writeHiveTable(sqlContext: SparkSession, tempDF: DataFrame,
                     outputTable: String, partition: String, hiveConf: String,
                     oldPartition: String, deleteTable: Boolean): Unit = {
    val schema = tempDF.schema
    val colNames = schema.fields.map(_.name).mkString(",")
    tempDF.createOrReplaceTempView("tmp_table")
    sqlContext.sql("SET hive.exec.dynamic.partition=true")
    sqlContext.sql("SET hive.exec.dynamic.partition.mode=nostrick")
    if (hiveConf != null) {
      sqlContext.sql(hiveConf)
    }
    val field = schema.fields.map {
      schema => schema.name + " " + (if (schema.dataType.typeName.equals("integer")) "int" else schema.dataType.typeName)
    }.mkString(",")

    println("fields:", field)

    if (deleteTable) {
      sqlContext.sql(s"drop table if exists $outputTable ")
    } else {
      sqlContext.sql(s"ALTER TABLE $outputTable DROP IF EXISTS PARTITION (dt='$oldPartition')")
    }
    sqlContext.sql(s"create table if not exists $outputTable($field) " +
      s"partitioned by (dt string) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' LINES TERMINATED BY '\n'"
    )

    //sqlContext.sql(s"alter table $outputTable drop if exists partition (dt=$partition)")

    println("sqlText:", s"insert overwrite table $outputTable partition(dt=$partition) SELECT $colNames FROM tmp_table")
    sqlContext.sql(s"insert overwrite table $outputTable partition(dt=$partition) SELECT $colNames FROM tmp_table")

  }

}
