File lists used for experiments in the SPIRE 2025 submission can be found in the directory ```Benchmarks```. 
File names found in these lists refer to files in the coli3682_dataset which can be found at the following url https://zenodo.org/records/6577997.
Before runinng the code these files should be preprocessed to remove the FASTA header line (i.e. the first line, which starts with '>') and all line breaks. 
Files preprocessed thus should then be concatenated with an ASCII '$' symbol between each. 
The first file of the concatenation is always used as the reference sequence for computing the Matching Statistics of the remaining files.

The executables are built by running ```make``` in the ```MS_GPU``` directory.
Experiments for a given file as preprocessed above are run by ```<executable name> <path to file>```.

For example, given that the file path is correct:
```sh
./lmp-ms ../Data/coli3682_dataset_150.txt
```

or

```sh
./ms-by-brackets-chunked ../Data/coli3682_dataset_150.txt
```

As an example of what preprocessing should achieve, given two files containing:

```
\>E.coli34
AACCAGATT
```
and
```
\>ec54443dd
ACAAAATA
```
should become a single file containing:

```AACCAGATT$ACAAAATA$```
