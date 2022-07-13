build:
	docker build -t indicadorbernardos_img .

up:
	docker run -t -d \
	-p 127.0.0.1:34617:34617/tcp \
	--name indicadorbernardos_cont \
	indicadorbernardos_img

run:
	[ -f "all_btc_tweets_GonBernardos.npy" ] && docker cp "all_btc_tweets_GonBernardos.npy" indicadorbernardos_cont:/srv/project
	docker exec -it indicadorbernardos_cont python3 indicador_bernardos.py
	docker cp -a indicadorbernardos_cont:/srv/project/output/ .

build_up_run:
	make build
	make up
	make run

down:
	docker stop indicadorbernardos_cont

rm_cont:
	docker rm indicadorbernardos_cont

rm_img:
	docker image rm indicadorbernardos_img

rm_all:
	make down
	make rm_cont
	make rm_img

rebuild:
	make rm_all
	make build
