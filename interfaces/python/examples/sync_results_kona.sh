#!/bin/bash


# Sync results remotely with kona64 server
rsync -avh \
    stellato@kona64.stanford.edu:/home/stellato/projects/osqp/interfaces/python/examples/results/ ./results/;
