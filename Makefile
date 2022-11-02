DOCKERFILE_SRC ?= docker/Dockerfile
DOCKER_BUILDNAME ?= yolov

.DEFAULT_GOAL := build

Dockerfile:
	cp $(DOCKERFILE_SRC) Dockerfile

.PHONY: build
build: Dockerfile
	docker build -t $(DOCKER_BUILDNAME) .

.PHONY: container
container:
	docker run -it --name $(DOCKER_BUILDNAME) --rm \
		--volume $(shell pwd):/home/phdenzel/yolov \
		--net=host $(DOCKER_BUILDNAME) \
		bash

.PHONY: clean
clean:
	rm -f Dockerfile

.PHONY: uninstall
uninstall:
	docker rmi yolov
