# Indicador Bernardos
Código diseñado para el cálculo y análisis del indicador Bernardos. El código permite extender el cálculo del indicador para otros usuarios de Twitter.


Artículo original del indicador Bernardos: https://osf.io/87brk/. 

Contacto con el autor: f0uri3r@protonmail.com


# Prepración del entorno
Para la descarga de tuits, es necesario tener una cuenta de [Twitter Developer](https://developer.twitter.com). En ésta, descargaremos las *consumer keys* (API Key and API Key Secret) desde nuestro Portal -> Projects & Apps -> (Nombre del proyecto) -> Keys and tokens -> Consumer Keys.

Teniendo las claves localizadas y el repositorio clonado, procederemos a alguna de las siguientes opciones.


## Opción 1: ejecución en Docker
Si se quiere ejecutar el script directamente, se debe de emplear el comando `make build_up_run`

Para finalizar la ejeución, se puede emplear el comando `make rm_all`

**Nota**: en esta versión no es posible visualizar el dashboard generado por la librería de *plotly*.


## Opción 2: ejecución sin Docker
Teniendo la versión 3 de Python, preferiblemente la 3.9, se puede decantar por la instalación de las versiones especificadas en el fichero de requerimientos de Docker `pip3 install -r requirements.txt` o el otro fichero de requerimientos que actualizaría a versiones más recientes `pip3 install -r requirements_no_docker.txt`.

Una vez instalado, ejecutar el script: `python3 indicador_bernardos.py`


## Ejemplo de ejecución
```
$ python3 indicador_bernardos.py 
all_btc_tweets_GonBernardos.npy file detected. Update? (y/n): y
Enter the path of consumer key txt file (press enter if not required): keys/API_key.txt
Enter the path of consumer key secret txt file (press enter if not required): keys/API_key_secret.txt
392 tweets descargados
578 tweets descargados
772 tweets descargados
965 tweets descargados
1157 tweets descargados
1343 tweets descargados
1537 tweets descargados
1729 tweets descargados
1924 tweets descargados
2117 tweets descargados
2312 tweets descargados
2507 tweets descargados
2701 tweets descargados
2895 tweets descargados
3092 tweets descargados
3143 tweets descargados
Number of estimated clusters : 28
Adding tweets to BTC price chart...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 105/105 [00:00<00:00, 119.09it/s]
```
