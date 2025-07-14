# lrf-ms
Tool for computing matching statistics that uses the longest repeated factor (LRF) heuristic

## Installation

```sh
git clone https://github.com/fmasillo/lrf-ms.git
cd lrf-ms
make
```

## Usage: 

This tool works on fasta files.

```sh
./lrf_ms -r reference_file -c collection_file [options]
  Options: 
    -o <output file> : basename for the output files (default: reference_collection)
    -b <buffer size> : size of the buffer for holding the temporary MS entries in MB (default: 100)
    -t <number of threads> : number of threads to use (default: 8)
    -h : print this help
```
