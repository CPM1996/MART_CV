#include "conos.hpp"

// Opciones para el parseador de línea de comandos, junto con su valor por defecto y una explicación
string keys =
    "{ fichero f   | <none> | Fichero de imagen o video para entrada }"
    "{ camara c    | 0 | Un entero indicando la camara a usar como entrada}"
    "{ calib       | calibracion.txt | fichero con los parámetros de calibracion }"
    "{ modelo m    | yolov4-conos-tiny.cfg | Fichero .cfg de la red neuronal }"
    "{ weights w   | yolov4-conos-tiny.weights | Fichero .weights de la red neuronal }"
    "{ umbral u    | 0.1 | Umbral de deteccion de conos }"
    "{ nms         | 0.4 | Umbral NMS para evitar solapamiento de objetos }"
    "{ focal       | | Introduzca esta opcion para estimar la longitud focal de la camara. Para ello coloque un cono a un metro de la camara }"
    "{ help  h     | | Imprime este mensaje de ayuda }";

int main(int argc, char** argv) {

	CommandLineParser parser(argc, argv, keys);

	// Ejecuta el parser de la línea de comandos
	parser = CommandLineParser(argc, argv, keys);
    parser.about("Este programa detectara conos FS, estimara su posicion y lo mostrara por pantalla");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    // Abre el fichero o dispositivo de entrada
	VideoCapture cap;
	if (parser.has("fichero"))
		cap.open(parser.get<String>("fichero"));
	else
		cap.open(parser.get<int>("camara"));
	if (!cap.isOpened()) {
		cout << "Error: no se pudo abrir el fichero o dispositivo" << endl;
		return -1;
	}

	// Lee los parámetros de calibración
	Param_calibracion parametros;
	if(leer_parametros(parser.get<String>("calib"), parametros) != 0){
		cout <<"Error: no se pudo abrir el fichero con los datos de calibración" << endl;
		return -1;
	}

	// Inicializa la red YOLO a partir de 
	// los ficheros .cfg y .weights
	Net red = readNet(parser.get<String>("modelo"), parser.get<String>("weights"));

	// Establecemos el uso de Nvidia CUDA para 
	// ejecutar la red cuando sea posible
	red.setPreferableBackend(DNN_BACKEND_CUDA);
	red.setPreferableTarget(DNN_TARGET_CUDA);

	// Leemos los umbrales de confianza y 
	// NMS de la línea de comandos
	float umbral_confianza = parser.get<float>("umbral");
	float umbral_NMS = parser.get<float>("nms");

	// Si se ha llamado a la calibración, se estima f 
	// y se guarda en el fichero de parámetros
	if(parser.has("focal")){
		parametros.f = calibrar_f(cap, red, parametros, umbral_confianza, umbral_NMS);

		cout << "Longitud focal: " << parametros.f << endl;

		ofstream fichero_param;
		string linea;
		fichero_param.open(parser.get<String>("calib"));
		if(!fichero_param.is_open()) return -1;

		linea = to_string(parametros.f) + '\n';
		fichero_param << linea;

		linea = to_string(parametros.a1) + '\n';
		fichero_param << linea;
		linea = to_string(parametros.b1) + '\n';
		fichero_param << linea;
		linea = to_string(parametros.c1) + '\n';
		fichero_param << linea;

		linea = to_string(parametros.a3) + '\n';
		fichero_param << linea;
		linea = to_string(parametros.b3) + '\n';
		fichero_param << linea;
		linea = to_string(parametros.c3) + '\n';
		fichero_param << linea;


		fichero_param.flush();
		fichero_param.close();
		return 0;

	}

	// Abrimos una ventana
	namedWindow("Detección de conos", WINDOW_NORMAL);

	Mat imagen;
	vector<Conos_detectados> detectados;
	vector<Conos_localizados> localizados;

	clock_t tiempo, inicio = clock();
	int num_archivo = 0;
	int min_FPS = INT_MAX;

	// Bucle infinito
	while (1) {

		// Extraemos una nueva imagen. 
		// Si está vacía, terminamos el programa
		cap >> imagen;
		if (imagen.empty())
			break;

		// Instante temporal de referencia
		tiempo = clock();

		// Detectamosy estimamos los conos en la imagen
		detectados = detectar_conos(imagen, red, umbral_confianza, umbral_NMS);
		localizados = estimar_posicion(detectados, imagen, parametros);

		// Ticks que se ha tardado en procesar la imagen
		tiempo = clock() - tiempo;

		// Una iteración por cono para el dibujado
		for (int i = 0; i < detectados.size(); ++i) {

			// Asignamos un color al rectángulo y al texto
			// de cada cono según su tipo
			Scalar color;
			switch (detectados[i].tipo)
			{
			case 0:
				color = Scalar(255, 0, 0);		// Azul
				break;
			case 1:
				color = Scalar(0, 255, 255); 	// Amarillo
				break;
			case 2:
				color = Scalar(0, 128, 255); 	// Naranja
				break;
			case 3:
				color = Scalar(0, 0, 255); 		// Rojo
				break;
			default:
				color = Scalar(0, 0, 0);		// Negro
				break;
			}

			// Dibujamos el rectángulo
			rectangle(imagen, detectados[i].posicion, color);

			// Texto con la información de posición en metros.
			// Precisión de tres cifras (10cm cuando es mayor de 10m)
			std::ostringstream texto;
			texto.precision(3);
			texto << "x1: ";
			texto << localizados[i].x1;
			texto << "m x3: ";
			texto << localizados[i].x3;
			texto << "m";

			// Escribimos el texto en la imagen, encima del rectángulo
			std::string label = texto.str();
			Point label_loc = Point(detectados[i].posicion.x, detectados[i].posicion.y);
			putText(imagen, label, label_loc, FONT_HERSHEY_SIMPLEX, 0.3, color);

		}

		// Calculamos y escribimos los FPS y FPS mínimo
		int fps = CLOCKS_PER_SEC / (float)tiempo;
		int segs = (clock() - inicio)/CLOCKS_PER_SEC;
		if(fps < min_FPS && segs > 0) min_FPS = fps;
		ostringstream texto;
		texto << "FPS: ";
		texto << fps;
		texto << " MIN: ";
		texto << min_FPS;
		string label = texto.str();
		putText(imagen,label, Point(0,imagen.rows-1),FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 255));

		// Mostramos la imagen por pantalla
		imshow("Detección de conos", imagen);

		// Comprobamos entrada de teclado para salir 
		// o guardar imagen
		char caracter = waitKey(1);
		if(caracter == 'q') return 0;
		else if(caracter == ' '){ //Espacio

			// Fichero Resultado_x.png. Se pueden escribir varios por sesión
			string nombre_archivo = "Resultado_";
			nombre_archivo += to_string(num_archivo);
			nombre_archivo += ".png";
			num_archivo++;

			// Escribimos la imagen al archivo
			imwrite(nombre_archivo, imagen);
		}

	}
	waitKey();
	return 0;
}