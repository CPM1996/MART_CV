#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define PI 3.14159265

// Dimensiones de los conos en metros
const double anchura_cono = 0.228;
const double altura_cono = 0.325;
const double anchura_cono3 = 0.285;
const double altura_cono3 = 0.505;

using namespace cv;
using namespace dnn;
using namespace std;

// Una estructura con un cono detectado en la imagen
struct Conos_detectados {
	int tipo;				// 1: azul chico, 2: amarillo chico, 3: naranja chico, 4: naranja grande
	float confianza;		// Número entre 0 y 1 que indica la fiabilidad de la detección
	Rect posicion;			// Clase que contiene un rectángulo que envuelve al cono
};

// Una estructura con un cono localizado en el mundo real
struct Conos_localizados {
	int tipo;
	float confianza;
	float x3;				// Distancia en profundidad a la cámara en metros
	float x1;				// Distancia horizontal a la cámaa en metros
};

// Conjunto de parámetros para la estimación de posición
struct Param_calibracion {

	// Apertura focal, depende de la cámara y se ha 
	// de medir o estimar cuando se usa una cámara nueva
	double f;			

	// Ajuste polinómico de grado 2 
	// a x1 en función de x3
	double a1;
	double b1;
	double c1;

	// Ajuste polinómico de grado 2 a x3
	double a3;
	double b3;
	double c3;
};

// Lee los parámetros de calibración de un fichero con el siguiente formato:
//f
//a1
//b1
//c1
//a3
//b3
//c3

// Devuelve 0 si ha tenido éxito, -1 si ha fallado
int leer_parametros(string nombre_fichero, Param_calibracion &parametros);

// Función que detecta los conos contenidos en 
// una imagen usando una red YOLO ya inicializada
vector<Conos_detectados> detectar_conos(Mat &imagen, Net &red, float umbral_confianza, float umbral_NMS);

// Función que estima la posición de los conos 
// a partir de su posición en la imagen
vector<Conos_localizados> estimar_posicion(const vector<Conos_detectados> conos_detectados, const  Mat imagen, const Param_calibracion p);

// Función para estimar la apertura focal a partir 
// de un cono situado a un metro de la imagen
double calibrar_f(VideoCapture cap, Net red, Param_calibracion parametros, float umbral_confianza, float umbral_NMS);