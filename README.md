# KeSACNN
***
## Dependency package

KeSACNN uses the following dependencies:

- [Python 2.7](https://www.python.org/)
- [keras 2.0.3](https://keras.io/)
- [Tensorflow 1.1.0](https://www.tensorflow.org/)
- [numpy 1.12.1](http://www.numpy.org/)

## Content
- corpus
	- BioCreative II corpus
	- BioCreative III corpus

- src
	- attention_keras.py: the self-attention layer
	- BiLSTM.py: a BiLSTM baseline
	- CNN-Kim.py: a CNN baseline proposed by Kim 
	- SACNN.py: Our Self-Attention CNN model
	- KeSACNN-arc1.py: Our KeSACNN-I model
	- KeSACNN-arc2.py: Our KeSACNN-2 model

## Train a basic SACNN model
To train a basic SACNN model, you need to provide the file of the training set, testing set and word embedding, and run the SACNN.py script:

```
python SACNN.py  
```
## Train a KeSACNN-I model
To train our KeSACNN-I model, you need to provide the file of the training set, testing set, word embedding, BEN embedding and CUI embedding, and run the KeSACNN-arc1.py script:

```
python KeSACNN-arc1.py 
```
## Train a KeSACNN-II model
To train our KeSACNN-II model, you need to provide the file of the training set, testing set, word embedding, BEN embedding and CUI embedding, and run the KeSACNN-arc2.py script:

```
python KeSACNN-arc2.py 
```