# IDASH2021 Documentation Upon Submission:

## Team: EAGLYS Inc (KENMARO-RYOHEI)

We are KENMARO and RYOHEI, from EAGLYS Inc, at Tokyo, Japan.  
First of all, thank you for hosting this great competition.  
We appreciate all of your supports during competition and we have greatly enjoyed this competetion tasks.

**Upon submission, we will upload two implementations, Model1 and Model2.**

We will report following first,

## Measured time

- raw preprocessing time
- cipher inference time

## Accuracy for validation dataset

- accuracy using validation dataset compared with given labels

   
then, will explain the followings.
## Algorithm

- raw preprocessing algorithm
- cipher infenrece algorithm

## Differences of submitted two models

<br>

## Security Settings

<br>
Lastly, we will explain 

<br>
<br>


## How To Use Our Model

to desribe the commands need to run to run our model using docker image.  

<br>
<br>

## Measured time (using 2000 validation datasets)

|model\process|  Preprocess [ms]|  Inference [ms]|
|----| ---- | ---- |
|Model1|  17222   |  1569 |
|Model2|  44953  |1942    |

NOTE: this is measured in our environment and can be varied in real environment.  
Model1 is faster in preprocessing and inference, however, model1 uses very naive preprocessing method as described in following.  
Model2 uses reasonalbe preprocessing method by understanding each nucleotide features, however, computationally little bit heavier, which will be mentioned later.  


<br>

## Achieved Accuracy (using 2000 validation datasets)

|model|  Accuracy |  
|----| ---- |
|Model1|  98.8  |
|Model2|  98.2  |

By simply using validation dataset out of give open dataset,  
model1 and model2 shows almost same accuracy.  

<br>
<br>

## Algorithm

## Preprocessing

We followed two steps for preprocessing.

- Onehot encoding 
- Feature extraction by PCA

Since Model1 and Model2 have different approach to this Onehot encoding,  
we separated our submissions.  
We will explain the differences in more detail later.  

<br>

## Inference

We used 

- Logistic Regression


In order to achieve logistic regression inference with homomorphic encryption,  
we utilized Microsoft SEAL for our lattice based encryption.  
In addtion to the original implementation of SEAL, however,   

**we used our original trick to speed up the linear operation of logistic regression.**

<br>

Let us explain about this in little detail.  

<br>

### Speed point of view

**Instead of using canonical embedding by CKKS encoding, we used coefficient packing.**  

In Microsoft SEAL, only BFV scheme has coefficient packing which is the direct encoding the number onto polynomial coefficient, CKKS only supports canonical embedded space in sub-ring to utilize SIMD operation.  
This is very powerful in some cases because summation and multiplication in each sub-ring can be easily done.  
However, in order to do inner product of two vector, rotation operation is needed to sum up the element between the sub-ring.  

**Since we found this inner product can be done more efficiently using coefficient method, 
we implemeted this coeffifient encoding method in CKKS scheme 
at https://github.com/kenmaro3/SEAL/tree/ckks_coeff_365** 

By using this coefficient for CKKS, we can have advantages for speed significantly.  
Since we can pack two vectors in forward-backward order to calculate inner product of two which uses intrinsic convolutional operation of two polynomial, 
we do not have to operate rotation operation in the case of CKKS canonical embedded encoding.  

For example, we want to calculate inner product of two vectors,   
$W = {2, 4, 1}$, $X = {1, 2, 2}$,  

then these vectors are embedded into ploynoinal space as:
```
X = x^2 + 2x + 2
W = x^2 + 4x + 2
```

Note that the weights are embedded in reversed order.
Then multiply these polynominals, you can get

```
x^4 + 6x^3 + 12x^2 + 12x + 4
```
where the coefficient of x^2 is the result of dot product.


Since the rotation operation in CKKS is as heavy as ciphertext multiplication, cutting this operation can reduce the computation time significantly.  

Not only cutting rotation operation, this method leads us to reduce the modulus chain of the encryption settings, hence, this method can benefit us using lighter encryption parameters, which can also contribute to improve the speed for encryption, decryption and computation.  
For this reasons, we decided to use CKKS + Coefficient method in our solutions.  

<br>

### Accuracy point of view
Since we use CKKS scheme, we can achieve better accuracy than using BFV. 
We can expect more flexible implementation for real number encoding, which leads us not to degrade the inference accuracy compared with raw inference as best as possible.  

In conclusion, our coefficient encoding method using CKKS holds the advantage as described table bellow.  


<br>

### Summary of Our Approach


|Scheme\Point of View|  Accuracy |  Speed |
|----| ---- | ---- |
|CKKS + Cannonical embedding| [x] |  [ ]|
|BFV  + Coefficient encoding|  [ ] | [x]|
|CKKS + Coefficient encoding (ours)|  [x] | [x]|

Using CKKS enables us to embed real number, which has huge advantage over BFV in this ML application.  

CKKS Coefficient encoding method can achieve **10x speed up** compared with CKKS Cannonical embedding encoding method, which was measured by Ryohei.  
In his implementation (Model2), this significance can be tested quite easily by switching two function in main program ON and OFF.  

```
int main{
    ...
    ...
    // if you want to run with basic lr method please use this function
    main_batch(parameter_folder, input_path, output_path);
    // if you want to use coefficient encoding lr pelase use this function
    // main_coef(parameter_folder, input_path, output_path);
}

```

<br>
<br>
<br>

## Diffences between Model1 and Model2

<br>

### Model1
The differences between model1 and model2 is the approach to the preprocessing.  
In short, model1 uses more naive approach to cut the length of input sequence.  

When model1 reads the input sequence, it only reads every 50 protein,  
therefore, when it reads the sequence, each sequence becomes length of 600.  
After this loading, unique value for each column is found by Pandas library and converted to one-hot form.  
As a result, one hot encoded sequence becomes around 3000 dimensions, 
we take this as an input for PCA, which brings dimension down to 200.  

We employed this approach in the beggining of implementation since MASH uses hash function for chunked sequence.  
We tried to mimick this approach naively first to see how much accuracy we can still achieve.  
Actually we found that this naive can already achieve 98.8% for validation datasets as reported above.  

<br>

### Model2

Instead of reading not all the protein from the sequence,  
model2 reads all the protein of the sequence.  
If we naively extend each column into all the available nucleotide, one-hot encoding becomes too large.
To avoid above and to improve speed, input RNAs are embedded into the 4dim existance ratio vector as following map function.
This can be done because all the nucleotide alphabet can be described by lienar superposition of basic nucleotide, which is A, C, G and T.  

```
nucleotide_map = {
  #     A    C    G    T
  "A": [1,   0,   0,   0],
  "C": [0,   1,   0,   0],
  "G": [0,   0,   1,   0],
  "T": [0,   0,   0,   1],
  "R": [0.5, 0,   0.5, 0],   # A or G
  "Y": [0,   0.5, 0,   0.5], # C or T
  "S": [0,   0.5, 0.5, 0],  # G or C
  "W": [0.5, 0,   0,   0.5],  # A or T
  "K": [0,   0,   0.5, 0.5],  # G or T
  "M": [0.5, 0.5, 0,   0],  # A or C
  "B": [0,   0.33,0.33,0.33],  # C or G or T
  "D": [0.33,0,   0.33,0.33],  # A or G or T
  "H": [0.33,0.33,0,   0.33],  # A or C or T
  "V": [0.33,0.33,0.33,0],  # A or C or G
  "N": [0.25,0.25,0.25,0.25],  # any base
  "O": [0,0,0,0],  # any base
}
```

For example, simple sequence such as ATR can be embedded like this,  

```
embedding_func("ATR") = [1, 0, 0, 0, 0, 0, 0, 1, 0.5, 0, 0.5, 0]
```

Since this map each nucleotide to length of four,  
each sequence becomes around 100,000 dimension.  
This is taken as an input for PCA to extract the features down to 200 dimensions as model1.  

### Inference of model1 and model2

Both of the model1 and model2 uses Logistic Regression to classify 200 dimensions input to 4 dimensions of strain.  


<br>
<br>
<br>


## Security Settings

Microsoft SEAL with CKKS scheme is used for encryption scheme with 128bit security.
Parameters are as follows:

```
static const double SCALE = pow(2.0, 30);
static const int POLY_MODULUS_DEGREE = 4096;
static const std::vector<int32_t> MODULUS = {38, 30, 38};
```

## How To Use Our Model
<br>

## How To Use Model1

git repository is here (https://github.com/kenmaro3/idash2021)

<br>

### pull docker image from docker hub
```
$ docker pull kenmaro/idash2021
```

### run docker container
```
$ docker run --name idash -itd -v <path_to_the_folder_containing_fastafile>:/data/ kenmaro/idash2021

```
### enter docker container and run the inference
```
$ docker exec -it idash2021 bash
```

with this, you should see fasta file at /data/***.fa

and please make sure the path of fasta file at /from_local/run_test.sh 
line 3 is pointing to the file mounted to this container. (default /data/Challenge.fa)
at line 5, please specify the input datasize (# of test data, as default,  set as 2000)  

```
>>>run_test.sh
rm -rf /from_local/results /from_local/pp_data
mkdir /from_local/results /from_local/pp_data
python test_main.py /data/Challenge.fa
./seal/test_main_cpp /from_local/pp_data /from_local/trained_model /from_local/results 2000
```

### activate pyenv
```
$ source setup.sh  
```

### run run_test.sh
```
$ sh run_test.sh
```

### check results and labels
```
$ cat constants.py
```
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

<br>

## How To Use Model2




<br>
<br>

## Licences

Microsoft SEAL

```
MIT License

Copyright (c) Microsoft Corporation. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```