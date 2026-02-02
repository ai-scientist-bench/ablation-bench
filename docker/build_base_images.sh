#!/bin/bash

docker buildx build --platform linux/amd64 -t "ablations-bench:judge" -f judge.Dockerfile .

for name in $(ls ../data/papers/full)
do   
    docker buildx build --platform linux/amd64 -t "ablations-bench:$name" --build-context paper=../data/papers/full/$name -f base.Dockerfile .
done

for name in $(ls ../data/papers/without_ablations)
do   
    docker buildx build --platform linux/amd64 -t "ablations-bench:$name" --build-context paper=../data/papers/without_ablations/$name -f base.Dockerfile .
done