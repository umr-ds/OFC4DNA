#!/bin/bash

# download usearch from https://www.drive5.com/usearch/download.html into this folder.

for file in *.fasta; do
    if [ -e "$file" ]; then
        ./usearch11.0.667_i86linux32  -cluster_fast  "$file" -id 0.8 -centroids "centroid_${file}" -uc "clusters_${file}.uc"
    fi
done