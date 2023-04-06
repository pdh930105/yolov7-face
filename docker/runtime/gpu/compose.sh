export CONTAINERNAME="yoloface_runtime_2210"
xhost +
xhost +local:docker
echo $xhost
#docker-compose up -d --force-recreate --no-deps --build
mkdir $HOME/log
docker rm -f ${CONTAINERNAME}

docker-compose up -d --build
docker exec ${CONTAINERNAME} echo "xhost's display : "
docker exec ${CONTAINERNAME} echo $DISPLAY
docker exec ${CONTAINERNAME} echo "visualize test"
docker exec ${CONTAINERNAME} xclock
docker exec -it ${CONTAINERNAME} /bin/bash
