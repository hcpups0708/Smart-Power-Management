/**
  * Created by mhwong on 12/05/16.
  */
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random

object Classification {

  // data case class
  case class MatchData(id: String, month: Int, day: Int, hour: Int, first_usage: Double, second_usage: Double, third_usage: Double, target: Double)

  def parse(line: String) = {
    val pieces = line.split(',')
    val id = pieces(0)
    val month = pieces(1).toInt
    val day = pieces(2).toInt
    val hour = pieces(3).toInt
    val first_usage = pieces(4).toDouble
    val second_usage = pieces(5).toDouble
    val third_usage = pieces(6).toDouble
    val target = pieces(7).toDouble
    MatchData(id, month, day, hour, first_usage, second_usage, third_usage, target)
  }

  def main (args: Array[String]) {
    // spark configuration
    val conf = new SparkConf().setAppName("Smart Power Management").setMaster("local")
    // spark context
    val sc = new SparkContext(conf)
    // create sql context
    val sqlContext = new SQLContext(sc)

    // read nov data
    val nov_data = sc.textFile("file:///home/spark/mhwong/cloud/project/data/nov.csv").map(parse)
    nov_data.persist()

    // get train id list
    val train_id_list = nov_data.map(_.id).distinct().collect()

    // model list
    var model_list = new Array[LinearRegressionModel](train_id_list.size)

    // split training data and train model
    for(i <- train_id_list.indices) {
      val training_data = nov_data.filter(_.id == train_id_list(i)).map(data => LabeledPoint(data.target, Vectors.dense(data.first_usage, data.second_usage, data.third_usage)))
      val training_data_frame = sqlContext.createDataFrame(training_data)
      val lr = new LinearRegression()
      val model = lr.fit(training_data_frame)
      model_list.update(i, model)
    }


    // read dec data
    val dec_data = sc.textFile("file:///home/spark/mhwong/cloud/project/data/dec.csv").map(parse)
    dec_data.persist()

    // get test id list
    val test_id_list = dec_data.map(_.id).distinct().collect()

    // train
    for(i <- test_id_list.indices) {

      val testing_data = dec_data.filter(_.id == test_id_list(i)).map(data => LabeledPoint(data.target, Vectors.dense(data.first_usage, data.second_usage, data.third_usage)))
      val testing_data_frame = sqlContext.createDataFrame(testing_data)

      var model_id = 0
      // if exists a model for that id
      if(train_id_list.contains(test_id_list(i))) {
        model_id = i
      }
      else {
        model_id = Random.nextInt(train_id_list.length)
      }
      val model = model_list(model_id)

      val prediction = model.transform(testing_data_frame)

      // output result
      prediction.map(row => (row.getDouble(0), row.getDouble(2))).coalesce(1).saveAsTextFile("file:///home/spark/mhwong/cloud/project/result/" + test_id_list(i) + ".result")
    }
  }
}
