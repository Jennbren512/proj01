#!/bin/bash

# Output CSV file
OUTPUT="results_full.csv"

# Write header to CSV
echo "NUMT,NUMTRIALS,MegaTrialsPerSecond" > $OUTPUT

# Thread counts to test
THREADS=(1 2 4 6 8)

# Trial counts to test
TRIALS=(100 1000 10000 50000 100000)

# Loop through all combinations
for t in "${THREADS[@]}"
do
  for n in "${TRIALS[@]}"
  do
    echo "Running: NUMT=$t, NUMTRIALS=$n"
    g++ -DNUMT=$t -DNUMTRIALS=$n -DCSV -fopenmp proj01.cpp -o proj01
    ./proj01 >> $OUTPUT
  done
done
