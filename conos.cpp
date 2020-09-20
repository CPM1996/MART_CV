#include "conos.hpp"

int leer_parametros(string nombre_fichero, Param_calibracion &parametros){
	ifstream fichero_param;
	string linea;

	// Intenta abrir el fichero
	fichero_param.open(nombre_fichero);
	if(!fichero_param.is_open()) return -1;

	// Lee y convierte los parámetros línea a línea
	getline(fichero_param,linea);
	parametros.f = std::stod(linea,NULL);

	getline(fichero_param,linea);
	parametros.a1 = stod(linea,NULL);
	getline(fichero_param,linea);
	parametros.b1 = stod(linea,NULL);
	getline(fichero_param,linea);
	parametros.c1 = stod(linea,NULL);

	getline(fichero_param,linea);
	parametros.a3 = stod(linea,NULL);
	getline(fichero_param,linea);
	parametros.b3 = stod(linea,NULL);
	getline(fichero_param,linea);
	parametros.c3 = stod(linea,NULL);

	// Cierra el fichero
	fichero_param.close();

	return 0;

}

vector<Conos_detectados> detectar_conos(Mat &imagen, Net &red, float umbral_confianza, float umbral_NMS){
	// Crea un blob de entrada a la red. Su resolución ha de ser igual o mayor a la del fichero .cfg
	Mat blob = blobFromImage(imagen, 1.0, Size(416, 416), Scalar(), true, false, CV_8U);

	// Mete el blob normalizado en la red y la ejecuta
	red.setInput(blob, "", 0.00392, 0);
	vector<Mat> res;
	red.forward(res,red.getUnconnectedOutLayersNames());

	vector<Conos_detectados> conos;
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	// Una iteración por cada matriz devuelta. 
	// Por norma general, será una sola
	for (int i = 0; i < res.size(); ++i) {

		// Puntero a una fila de la matriz
		float* datos = (float*)res[i].data;

		// Recorre la matriz por filas
		for (int j = 0; j < res[i].rows; ++j, datos += res[i].cols) {

			// Un vector con una probabilidad por cada objeto
			Mat scores = res[i].row(j).colRange(5, res[i].cols);
			Point classIdPoint;
			double confidence;

			// Devuelve el número de objeto con mayor 
			// probabilidad, y dicha probabilidad
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

			// Si la probabilidad no supera el umbral, se descarta
			if (confidence > umbral_confianza)
			{

				// Convierte las coordenadas de flotantes 
				// normalizados a enteros medidos en píxeles
				int centerX = (int)(datos[0] * imagen.cols);
				int centerY = (int)(datos[1] * imagen.rows);
				int width = (int)(datos[2] * imagen.cols);
				int height = (int)(datos[3] * imagen.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				// Añade el nuevo cono a un grupo de vectores
				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Ejecuta el algoritmo NMS para eliminar duplicados
	vector<int> indices;
	NMSBoxes(boxes, confidences, umbral_confianza, umbral_NMS, indices);

	// Añade los conos no eliminados al vector de conos detectados
	Conos_detectados cono;
	for (int i = 0; i < indices.size(); ++i)
	{
		int indice = indices[i];
		cono.tipo = classIds[indice];
		cono.confianza = confidences[indice];
		cono.posicion = boxes[indice];
		conos.push_back(cono);
	}
	return conos;
}

vector<Conos_localizados> estimar_posicion(const std::vector<Conos_detectados> conos_detectados, const  Mat imagen, const Param_calibracion p) {
	vector<Conos_localizados> res;
	Conos_detectados cono;
	Conos_localizados estimacion;

	double x1, x3, y1, y2, ancho_y, alto_y;

	// Una iteración por cono
	for (size_t i = 0; i < conos_detectados.size(); ++i) {

		// Copiamos los datos comunes de una estructura a otra
		cono = conos_detectados[i];
		estimacion.tipo = cono.tipo;
		estimacion.confianza = cono.confianza;

		// Dimensiones normalizadas a una imagen de altura 1
		ancho_y = (double) cono.posicion.width / imagen.rows;
		alto_y = (double) cono.posicion.height / imagen.rows;
		y1 = (double) (cono.posicion.x + cono.posicion.width / 2.0 - imagen.cols/2.0) / imagen.rows;
		y2 = (double) (cono.posicion.y + cono.posicion.height / 2.0) / imagen.rows - 0.5;

		// Como el cono narnaja grande tiene una dimensiones distintas, se calcula aparte
		if (cono.tipo == 3) {
			x1 = y1 * anchura_cono3 / ancho_y;
			x3 = p.f * (anchura_cono3 / ancho_y + altura_cono3 / alto_y) / 2;
		}
		else {
			x1 = y1 * anchura_cono / ancho_y;
			x3 = p.f * (anchura_cono / ancho_y + altura_cono / alto_y) / 2;
		}

		// Ajuste polinómico de las medidas
		estimacion.x3 = p.c3*std::pow(x3,2) + p.b3*x3 + p.a3;
		estimacion.x1 = x1 + p.c1*std::pow(x3,2) + p.b1*x3 + p.a1;;
		res.push_back(estimacion);
	}
	return res;
}

double calibrar_f(VideoCapture cap, Net red, Param_calibracion parametros, float umbral_confianza, float umbral_NMS){
	Mat imagen;
	vector<Conos_detectados> detectados;
	Conos_detectados cono;
	double f;

	// Ejecuta la red neuronal hasa que encuentre un cono
	do{
			cap >> imagen;
			if (imagen.empty()){
				cout << "Error: no se puede obtener la imagen o video" << endl;
				return -1;
			}
			detectados = detectar_conos(imagen, red, umbral_confianza, umbral_NMS);
	}while(detectados.size() == 0);

	// Utiliza el primer cono encontrado para hacer la estimación
	// Se presupone un solo cono
	cono = detectados[0];
	
	// Se normaliza igual que en la función estimar_posicion
	double ancho_y = (double) cono.posicion.width / imagen.rows;
	double alto_y = (double) cono.posicion.height / imagen.rows;

	// Se estima f
	if (cono.tipo == 3) {
		
		f = 2.0 /  (anchura_cono3 / ancho_y + altura_cono3 / alto_y);
	}
	else {
		f = 2.0 /  (anchura_cono / ancho_y + altura_cono / alto_y);
	}
	return f;
}