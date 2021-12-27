#!/bin/bash

set -e

for i in 26 24 25 23; do
	docker-compose run --rm notebooks ipython3 experiments/karnataka.py -- --village vil${i}
done

