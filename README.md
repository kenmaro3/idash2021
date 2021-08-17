# Document for usage:

### Team: KENMARO-YAMAPOKE


We submit 2 implementations, basics of algorithm is same for both of them,
the difference between them is the preprocessing part.
Model 1 is a little bit more naive preprocessing than model 2.

Please evaluate both of the models following instructions bellow.


## Model1

We provide decryption and usage of this model.

## Description
This model uses
- One hot encoding for the input sequence
- PCA to reduce/extract features (down to 200)
- Logistic regression for classification (200 -> 4)

## Usage
### git clone repo
$ git clone https://github.com/kenmaro3/idash2021

### build docker image
$ cd idash2021 
$ docker build -t idash . 

### untar trained model under idash2021
$ cd idash2021 
$ tar -xzvf  trained_model.tar.gz 

### run docker container
$ docker run --name idash2021 -itd -v ./idash2021:/from_local

### enter docker container and run the inference
$ docker exec -it idash2021 bash

### look at /from_local/run_test.sh
$ cat run_test.sh

```
>>>run_test.sh
rm -rf /from_local/results /from_local/pp_data
mkdir /from_local/results /from_local/pp_data
source ~/.pyenv/versions/myenv/bin/activate
python test_main.py /from_local/Challenge/Challenge.fa
./seal/test_main_cpp /from_local/pp_data /from_local/trained_model /from_local/results 2000
```

at line 4, you can specify the input fasta file. 
as default, it is set as /from_local/Challenge/Challenge.fa 
at line 5, please specify the input datasize (# of test data, as default,  set as 2000)

### run run_test.sh
$ sh run_test.sh

### check results and labels
$ cat constants.py
```
xs = [">B.1.427", ">B.1.1.7", ">P.1", ">B.1.526"]
```
1st line of constants.py describes the label of the results. 
e.g) if label is 0, means that sequence is >B.1.427.

$ ls -l /from_local/results
```
-rw-r--r-- 1 root root    28 Aug 17 05:09 computation.csv
-rw-r--r-- 1 root root    27 Aug 17 05:09 decryption.csv
-rw-r--r-- 1 root root    27 Aug 17 05:09 encryption.csv
-rw-r--r-- 1 root root 96244 Aug 17 05:09 label.csv
-rw-r--r-- 1 root root 89033 Aug 17 05:09 probability.csv
-rw-r--r-- 1 root root  4000 Aug 17 05:09 result_cipher_label.txt
-rw-r--r-- 1 root root    26 Aug 17 05:09 roundtrip.csv
```

### description of each output files

- computation.csv: time for server side computation
- decryption.csv:    time for client side decryption
- encryption.csv:    time for client side encryption
- label.csv:             columns is index, id, class (index is aligned with probability.csv and result_cipher_label.csv, id is id from fasta file, class is index of the strain)
- probability.csv:     probabilities of each DNA sequence (order is same with index of label.csv)
- roundtrip.csv :      time for entire cipher classification
