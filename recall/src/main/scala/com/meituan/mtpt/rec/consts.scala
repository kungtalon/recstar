package com.meituan.mtpt.rec

import org.apache.spark.sql.DataFrame

object consts {

  def broadTable(mode: String) : String = {
      s"""
         |select t_main.order_id order_id,
         |       user_id,
         |       t_main.product_id product_id,
         |       add_to_cart_order,
         |       reordered,
         |       order_number,
         |       order_dow,
         |       order_hour_of_day order_hod,
         |       days_since_prior_order,
         |       aisle_id,
         |       department_id
         |  from ba_dealrank.recommend_star_order_products__$mode t_main
         |  left join ba_dealrank.recommend_star_orders t_order
         |    on t_order.order_id = t_main.order_id
         |  left join ba_dealrank.recommend_star_products t_prod
         |    on t_main.product_id = t_prod.product_id
         |""".stripMargin
  }

  def itemHotTime() : String = {
      s"""
         |select *
         |  from (
         |        select *,
         |               row_number() over (partition by product_id order by order_cnt desc) as ranking
         |          from (
         |                select count(t_main.order_id) order_cnt,
         |                       t_main.product_id product_id,
         |                       order_dow,
         |                       order_hour_of_day order_hod
         |                  from ba_dealrank.recommend_star_order_products__prior t_main
         |                  left join ba_dealrank.recommend_star_orders t_order
         |                    on t_order.order_id = t_main.order_id
         |                 group by product_id,
         |                          order_dow,
         |                          order_hour_of_day
         |               ) t
         |       ) a
         | where ranking <= 1
         |
         |""".stripMargin

  }

  def itemOrderIndex(): String ={
      s"""
         |select a.product_id product_id,
         |       order_pv,
         |       order_uv,
         |       reorder_pv,
         |       reorder_uv
         |  from (
         |        select count(t_train.order_id) order_pv,
         |               sum(reordered) reorder_pv,
         |               count(distinct user_id) order_uv,
         |               t_train.product_id product_id
         |          from ba_dealrank.recommend_star_order_products__train t_train
         |          left join ba_dealrank.recommend_star_orders t_order
         |            on t_order.order_id = t_train.order_id
         |         group by product_id
         |       ) a
         |  left join (
         |        select count(distinct user_id) reorder_uv,
         |               t_train.product_id product_id
         |          from ba_dealrank.recommend_star_order_products__train t_train
         |          left join ba_dealrank.recommend_star_orders t_order
         |            on t_order.order_id = t_train.order_id
         |         group by product_id
         |       ) b
         |    on a.product_id = b.product_id
         |
         |""".stripMargin

  }

  def itemAvgCartOrder(): String ={
      s"""
         |select product_id product_id,
         |       avg(add_to_cart_order) avg_cart_order
         |  from ba_dealrank.recommend_star_order_products__prior
         | group by product_id
         |""".stripMargin
  }

  def allOrders(): String = {
    s"""
       |select orders.order_id,
       |       user_id,
       |       product_id,
       |       order_number,
       |       add_to_cart_order,
       |       reordered
       |  from ba_dealrank.recommend_star_orders orders
       |  left join ba_dealrank.recommend_star_order_products__prior `prior`
       |    on orders.order_id = prior.order_id
       | where eval_set = 'prior'
       | union select orders.order_id,
       |       user_id,
       |       product_id,
       |       order_number,
       |       add_to_cart_order,
       |       reordered
       |  from ba_dealrank.recommend_star_orders orders
       |  left join ba_dealrank.recommend_star_order_products__train `train`
       |    on orders.order_id = train.order_id
       | where eval_set = 'train'
       |""".stripMargin
  }

  def mainOrdersContext(evalSet:String): String = {
    s"""
       |select orders.order_id,
       |       order_dow,
       |       order_hour_of_day,
       |       days_since_prior_order
       |  from ba_dealrank.recommend_star_orders orders
       |  left join ba_dealrank.recommend_star_order_products__$evalSet `main`
       |    on orders.order_id = main.order_id
       | where eval_set = '$evalSet'
       |""".stripMargin
  }

}
