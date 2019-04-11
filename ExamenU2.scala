/*Desarrollar de las siguientes instrucciones en Spark con el lenguaje de programacion Scala,
 utilizando solo la documentacion de la libreria Machine Learning ML de Spark y google.*/
/*Del dataset llamado "Iris.csv" que se encuentra en github */
import org.apache.spark.sql.SparkSession
import spark.implicits._
import org.apache.spark.sql.Column
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

//Inicio de el archivo de Iris.csv
val spark = SparkSession.builder.master("local[*]").getOrCreate()
//Limpieza de los datos
val df = spark.read.option("inferSchema","true").csv("Iris.csv").toDF("SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "class")

val newcol = when($"class".contains("Iris-setosa"), 1.0).
  otherwise(when($"class".contains("Iris-virginica"), 3.0).
  otherwise(2.0))

val newdf = df.withColumn("etiqueta", newcol)

newdf.select("etiqueta","SepalLength","SepalWidth","PetalLength","PetalWidth","class").show(150, false)

/*a. Utilice el algoritmo de Machine Learning llamado multilayer perceptron que viene en la libreria de Spark*/
//Juntando el data
val assembler = new VectorAssembler().setInputCols(Array("SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "etiqueta")).setOutputCol("features")
//Transformacion los data en features
val features = assembler.transform(newdf)
features.show(5)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Acomodar los labels, asi para añadir metadata para la columna label.
val labelIndexer = new StringIndexer().setInputCol("class").setOutputCol("indexedLabel").fit(features)
println(s"Found labels: ${labelIndexer.labels.mkString("[", ", ", "]")}")

//Se hace una categorizacion de las caracterisiticas de features y de ahi de indexan
val featureIndexer = new VectorIndexer().setInputCol("features")
.setOutputCol("indexedFeatures")
.setMaxCategories(4)
.fit(features)


/*Se seleccionó un 60% de entrenamiento y un 40% de prueba 
Guardamos nuestras separaciones de entrenamiento y de prueba*/
val splits = features.randomSplit(Array(0.6, 0.4))
val trainingData = splits(0)
val testData = splits(1)

//Para realizar la predicción usando MLP importamos las librerías del clasificador MLP así como su evaluador.//

/*b. Disene su propia arquitectura con un minimo de tres neuronas de entrada, dos capas en la capa oculta con mas de tres neuronas
cada una y finalmente dos o mas neuronas en la capa de salida*//* La función para calcular los nuevos pesos fue (w<t+1> = w + b(error)(z)) que
nos ayuda a asignar nuevos pesos para la siguiente iteración, la fórmula para
calcular el error por predeterminado es un algoritmo de optimización llamado
L-BFGS usando una limitada cantidad de memoria, se usa para estimación de
parámetros.*/
// La capa de entrada será 4 (4 características), dos nodos escondidos de 4 cada uno y 4 de salida (clases)
val layers = Array[Int](4, 4, 4, 4)

/*c. EXplique detalladamente cada paso del proceso Machine Learning dentro del codigo que desarrolle*/
// Se crea el entrenador y se asignan los parámetros
//.setLayers(layers) se usa para asignar las capas que se crearon previamente (5 de entrada, dos intermedias de 4 y 4 de salida)
//.setSeed es la asignación aleatoria de la semilla que indica los pesos iniciales en caso de no haber sido asignados.
//.setMaxIter es el número máximo de iteraciones por la cual se realizarán los cálculos.
//.setBlockSize es el tamaño del bloque para poner los datos de entrada en matrices que nos ayudan a agilizar los cálculos
val trainer = new MultilayerPerceptronClassifier()
.setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128)
.setSeed(System.currentTimeMillis)
.setMaxIter(200)

// Entrenamos el modelo
val model = pipeline.fit(trainingData)
// Calculamos la precisión de nuestro modelo
val predictions = model.transform(testData)
predictions.show(5)

//A nuestro evaluador se le asignará la métrica de precisión, de ésta manera podremos saber qué tan preciso es nuestro modelo
val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("indexedLabel")
.setPredictionCol("prediction")
.setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))


//d. Explique detalladamente la funcion matematica de entrenamiento que utilizo, con sus propias palabras//
/* La función matemática que se implementa de manera predeterminada en los
nodos intermedios es la función sigmoide (1/1+e^-n), se utiliza como función de
activación que nos ayuda con datos binarios o múltiples mientras que en los
nodos de salida se utiliza la función softmax (e^(xi) / sum(e^(zk) )) que nos
ayuda con la clasificación múltiple así como los cálculos de predicciones para
las mismas*/

//e. Explique la funcion de error que utilizo para el resultado final//
/* La función para calcular los nuevos pesos fue (w<t+1> = w + b(error)(z)) que
nos ayuda a asignar nuevos pesos para la siguiente iteración, la fórmula para
calcular el error por predeterminado es un algoritmo de optimización llamado
L-BFGS usando una limitada cantidad de memoria, se usa para estimación de
parámetros.*/.

/*f. Finalmente suba el codigo a github y documente detalladamente su codigo asi como sus resultados*/
