DOCKERFILE_SRC ?= docker/Dockerfile
DOCKER_BUILDNAME ?= yolov

.DEFAULT_GOAL := build

Dockerfile:
	cp $(DOCKERFILE_SRC) Dockerfile

.PHONY: build
build: Dockerfile
	docker build -t $(DOCKER_BUILDNAME) .

.PHONY: container
container: build
	docker run -it --name $(DOCKER_BUILDNAME) --rm \
		--volume $(pwd):/home/phdenzel/yolov \
		--net=host $(DOCKER_BUILDNAME) \
		sh

.PHONY: clean
clean:
	rm -f Dockerfile

.PHONY: uninstall
uninstall:
	docker rmi yolov
