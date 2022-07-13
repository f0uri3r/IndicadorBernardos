# Indicador Bernardos
Código diseñado para el cálculo y análisis del indicador Bernardos. El código permite extender el cálculo del indicador para otros usuarios de Twitter.


Artículo original del indicador Bernardos: https://osf.io/87brk/. 

Contacto con el autor: f0uri3r@protonmail.com


# Prepración del entorno
Para la descarga de tuits, es necesario tener una cuenta de [Twitter Developer](https://developer.twitter.com). En ésta, descargaremos las *consumer keys* (API Key and Secret) desde nuestro Portal -> Projects & Apps -> (Nombre del proyecto) -> Keys and tokens -> Consumer Keys.

Teniendo las claves localizadas y el repositorio clonado, procederemos a alguna de las siguientes opciones.


## Opción 1: ejecución en Docker
Si se quiere ejecutar el script directamente, se debe de emplear el comando `make build_up_run`

Para finalizar la ejeución, se puede emplear el comando `make rm_all`

**Nota**: en esta versión no es posible visualizar el dashboard generado por la librería de *plotly*.

## Opción 2: ejecución sin Docker
Teniendo la versión 3 de Python, preferiblemente la 3.9, se puede decantar por la instalación de las versiones especificadas en el fichero de requerimientos de Docker `pip3 install -r requirements.txt` o el otro fichero de requerimientos que actualizaría a versiones más recientes `pip3 install -r requirements_no_docker.txt`.

Una vez instalado, ejecutar el script: `python3 indicador_bernardos.py`
