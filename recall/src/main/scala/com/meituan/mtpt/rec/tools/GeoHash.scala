package com.meituan.mtpt.rec.tools

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object GeoHash {

  val base32 = "0123456789bcdefghjkmnpqrstuvwxyz"

  def decodeBounds( geohash:String ):((Double,Double),(Double,Double)) = {
    def toBitList( s:String ) = s.flatMap{
      c => ("00000" + base32.indexOf(c).toBinaryString ).
        reverse.take(5).reverse.map('1' == ) } toList

    def split( l:List[Boolean] ):(List[Boolean],List[Boolean]) ={
      l match{
        case Nil => (Nil,Nil)
        case x::Nil => ( x::Nil,Nil)
        case x::y::zs => val (xs,ys) =split( zs );( x::xs,y::ys)
      }
    }

    def dehash( xs:List[Boolean] , min:Double,max:Double):(Double,Double) = {
      ((min,max) /: xs ){
        case ((min,max) ,b) =>
          if( b )( (min + max )/2 , max )
          else ( min,(min + max )/ 2 )
      }
    }

    val ( xs ,ys ) = split( toBitList( geohash ) )
    ( dehash( ys ,-90,90) , dehash( xs, -180,180 ) )
  }

  def decode( geohash:String ):(Double,Double) = {
    decodeBounds(geohash) match {
      case ((minLat,maxLat),(minLng,maxLng)) => ( (maxLat+minLat)/2, (maxLng+minLng)/2 )
    }
  }

  def encode( lat:Double, lng:Double ):String = encode(lat,lng,12)
  def encode( lat:Double, lng:Double, precision:Int ):String = {

    var (minLat,maxLat) = (-90.0,90.0)
    var (minLng,maxLng) = (-180.0,180.0)
    val bits = List(16,8,4,2,1)

    (0 until precision).map{ p => {
      base32 apply (0 until 5).map{ i => {
        if (((5 * p) + i) % 2 == 0) {
          val mid = (minLng+maxLng)/2.0
          if (lng > mid) {
            minLng = mid
            bits(i)
          } else {
            maxLng = mid
            0
          }
        } else {
          val mid = (minLat+maxLat)/2.0
          if (lat > mid) {
            minLat = mid
            bits(i)
          } else {
            maxLat = mid
            0
          }
        }
      }}.reduceLeft( (a,b) => a|b )
    }}.mkString("")
  }

  def adjacent(geohash:String, direction:String):String =  {
    val neighbour = mutable.HashMap(
      "n" -> Array("p0r21436x8zb9dcf5h7kjnmqesgutwvy", "bc01fg45238967deuvhjyznpkmstqrwx"),
      "s" -> Array("14365h7k9dcfesgujnmqp0r2twvyx8zb", "238967debc01fg45kmstqrwxuvhjyznp"),
      "e" -> Array("bc01fg45238967deuvhjyznpkmstqrwx", "p0r21436x8zb9dcf5h7kjnmqesgutwvy"),
      "w" -> Array("238967debc01fg45kmstqrwxuvhjyznp", "14365h7k9dcfesgujnmqp0r2twvyx8zb")
    )

    val border =  mutable.HashMap(
      "n" -> Array("prxz", "bcfguvyz"),
      "s" -> Array("028b", "0145hjnp"),
      "e" -> Array("bcfguvyz", "prxz"),
      "w" -> Array("0145hjnp", "028b")
    )

    val lastCh = geohash.charAt(geohash.length-1)    // last character of hash
    var parent = geohash.take(geohash.length-1) // hash without last character
    val geo_type = geohash.length % 2
    if (border(direction)(geo_type).indexOf(lastCh) != -1 && parent != "") {
      parent = adjacent(parent, direction)
    }

    parent + base32.charAt(neighbour(direction)(geo_type).indexOf(lastCh))
  }

  def neighbours(hash_value: String):ArrayBuffer[String] =  {
    val neighbors = new ArrayBuffer[String]()

    neighbors.append(adjacent(hash_value, "n"))
    neighbors.append(adjacent(adjacent(hash_value, "n"), "e"))
    neighbors.append(adjacent(hash_value, "e"))
    neighbors.append(adjacent(adjacent(hash_value, "s"), "e"))
    neighbors.append(adjacent(hash_value, "s"))
    neighbors.append(adjacent(adjacent(hash_value, "s"), "w"))
    neighbors.append(adjacent(hash_value, "w"))
    neighbors.append(adjacent(adjacent(hash_value, "n"), "w"))

    neighbors
  }

  def main(args: Array[String]): Unit = {
    println(neighbours("wwymk"))
    println(adjacent("wwymkmg", "n"))
    println(decode("w"))
    println(encode(23.08502197265625,114.4061279296875,1))
    println(encode(23.08502197265625,114.4061279296875,2))
    println(encode(23.08502197265625,114.4061279296875,3))
    println(encode(23.08502197265625,114.4061279296875,4))
    println(encode(23.08502197265625,114.4061279296875,5))
    println(encode(23.08502197265625,114.4061279296875,6))
    println(encode(23.08502197265625,114.4061279296875,12))

  }
}