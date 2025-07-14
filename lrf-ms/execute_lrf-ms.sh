#!/bin/bash
set -u

echo "Text file : $1";
echo "Max number of threads: $2";

for (( pt=22; pt<="$2"; pt++ ))
do
  ./lrf_ms -z  "$1" -t "$pt"
done

