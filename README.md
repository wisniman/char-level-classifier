Character-level Document Classification with CNN and RNN/LSTM
===
## Objective
Comparison between following architectures for document classification:
1. __Convolutional Neural Networks:__
   - Character-level Convolutional Networks (char-CNN): [[Zhang and LeCun2015]](https://arxiv.org/abs/1502.01710), [[Zhang et al.2015]](https://arxiv.org/abs/1509.01626)
   - Very Deep Convolutional Networks for Text Classification (VDCNN): [[Conneau et al.2016]](https://arxiv.org/abs/1606.01781)
2. __Recurrent Neural Networks:__
   - Word-based (word2vec) LSTM: [[Zhang and LeCun2015]](https://arxiv.org/abs/1502.01710), [[Greff et al.2015]](https://arxiv.org/abs/1503.04069), [[Pascanu et al.2012]](https://arxiv.org/abs/1211.5063), [[Graves and Schmidhuber2005]](ftp://ftp.idsia.ch/pub/juergen/nn_2005.pdf)
   - Bidirectional LSTM: [[Graves and Schmidhuber2005]](ftp://ftp.idsia.ch/pub/juergen/nn_2005.pdf), + Max Pooling [[Zhou et al.2016]](https://arxiv.org/abs/1611.06639)
   - Stacked LSTM
   - _(Gated Recurrent Neural Network: [[Tang et al.2015]](http://aclweb.org/anthology/D15-1167))_
3. __Combined Convolutional and Recurrent Neural Networks:__
   - Convolutional Network and LSTM (char-CRNN): [[Xiao and Cho2016]](https://arxiv.org/abs/1602.00367), [[Zhou et al.2015]](https://arxiv.org/abs/1511.08630)
4. __Other:__
   - _Fasttext: [[Joulin2016]](https://arxiv.org/abs/1607.01759)_
   - Encoder-Decoder, Recurrent, Generative vs. Discriminative [[Yogatama et al.2017]](https://arxiv.org/abs/1703.01898), ...
---
## Architectures
### Convolutional Neural Networks
#### Character-level Convolutional Networks (char-CNN)
!["Text Understanding from Scratch: Figure 2. Illustration of our model"](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/1336146e7f95b295bb73c7659c6af4befd86cbdd/3-Figure2-1.png)

[Image source: [Semantic Scholar](https://www.semanticscholar.org/paper/Text-Understanding-from-Scratch-Zhang-LeCun/1336146e7f95b295bb73c7659c6af4befd86cbdd)]

#### Very Deep Convolutional Networks for Text Classification (VDCNN)
!["Very Deep Convolutional Networks for Text Classification. Figure 1: Global architecture with convolutional blocks. See text for details."](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/84ca430856a92000e90cd728445ca2241c10ddc3/3-Figure1-1.png)

[Image source: [Semantic Scholar](https://www.semanticscholar.org/paper/Very-Deep-Convolutional-Networks-for-Text-Classifi-Schwenk-Barrault/84ca430856a92000e90cd728445ca2241c10ddc3)]
### Recurrent Neural Networks
#### LSTM with Embedding Layer
#### LSTM with one-hot encoded input
...
#### Bi-directional LSTM with Embedding Layer
#### Bi-directional LSTM with one-hot encoded input
...
### Combined Convolutional and Recurrent Neural Networks
#### Convolutional Network and LSTM (char-CRNN)
...
### 1.4. Other
...

---
## Datasets
1. Predefined datasets:
   - __AG__'s news corpus _(English)_
   - __Sogou__ news corpus _(Chinese)_
   - __DBPedia__ ontology dataset _(English)_
   - __Yelp__ reviews _(English)_
   - __Yahoo! Answers__ dataset _(English)_
   - __Amazon__ reviews _(English)_
2. Self-generated dataset:
   - __Spiegel Online__ news corpus _(German)_

### Predefined Datasets
|Dataset                  |Train    |Test  |Classes|
|-------------------------|--------:|-----:|:-----:|
|AG's news                |120.000  |7.600 |4      |
|Sogou news               |450.000  |60.000|5      |
|DBPedia ontology         |560.000  |70.000|14     |
|Yelp reviews (Polarity)  |560.000  |38.000|2      |
|Yelp reviews (Full)      |650.000  |50.000|5      |
|Yahoo! Answers           |1.400.000|60.000|10     |
|Amazon reviews (Polarity)|3.600.000|650.000|2     |
|Amazon reviews (Full)    |3.000.000|400.000|5     |

### Self-generated Dataset
|Dataset                  |Train    |Test  |Classes|
|-------------------------|--------:|-----:|:-----:|
|Spiegel Online           |187.120  |20.745|5      |

#### Classes:
- __Politik:__ 138.859 articles
- __Sport:__ 89.524 articles
- __Panorama:__ 83.684 articles
- __Wirtschaft:__ 81.969 articles
- __Kultur:__ 44.270 articles
- _Wissenschaft: 29.443 articles_
- _Netzwelt: 28.977 articles_
- _Reise: 20.802 articles_
- _Leben: 16.409 articles_
- _Auto: 14.730 articles_
- _einestages: 7.568 articles_
- _Gesundheit: 5.160 articles_
- _KarriereSPIEGEL: 4.134 articles_
- _Stil: 709 articles_
- _Blogs: 1 articles_

### Generating Spiegel Online dataset
#### Scraping and Web Crawling Framework Scrapy
- [Scrapy](https://scrapy.org/), [Docs](https://docs.scrapy.org/en/latest/)
- [Scrapinghub](https://scrapinghub.com), [Docs](https://doc.scrapinghub.com/)
---
## Training on Google Cloud Machine Learning Engine
- [Cloud ML Engine Documentation](https://cloud.google.com/ml-engine/docs/)
- [Keras on Cloud ML Engine: MNIST Multi-Layer Perceptron](https://github.com/clintonreece/keras-cloud-ml-engine)

### Trainer Package setup
...
### Local training
``` shell
gcloud ml-engine local train ^
  --job-dir $JOB_DIR ^
  --module-name trainer.mnist_mlp ^
  --package-path ./trainer ^
  -- ^
  --train-file ./data/mnist.pkl
```
### Cloud training
``` shell
gcloud ml-engine jobs submit training $JOB_NAME ^
    --job-dir $JOB_DIR ^
    --runtime-version 1.0 ^
    --module-name trainer.mnist_mlp ^
    --package-path ./trainer ^
    --region $REGION ^
    -- ^
    --train-file gs://$BUCKET_NAME/data/mnist.pkl
```
### Hyperparameter Tuning
...

---
## References:
### Papers:
1. __[Conneau et al.2016]__ Alexis Conneau, Holger Schwenk, Loïc Barrault, and Yann LeCun. __[Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781)__. arXiv:1606.01781 [cs.CL], 2016.
- __[Gers2001]__ Felix A. Gers. __[Long Short-Term Memory in Recurrent Neural Networks](http://www.felixgers.de/papers/phd.pdf)__. PhD thesis, Department of Computer Science, Swiss Federal Institute of Technology, Lausanne, EPFL, Switzerland, 2001.
- __[Gers et al.2000]__ Felix A. Gers, Jürgen Schmidhuber, and Fred Cummins. __[Learning to Forget: Continual Prediction with
LSTM](https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf)__. In _Neural Computation_, Volume 12, Issue 10: 2451-2471, October 2000.
- __[Goldberg2015]__ Yoav Goldberg. __[A Primer on Neural Network Models for Natural Language Processing](https://arxiv.org/abs/1510.00726)__. arXiv:1510.00726 [cs.CL], 2015.
- __[Graves and Schmidhuber2005]__ Alex Graves and Jürgen Schmidhuber. __[Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures](ftp://ftp.idsia.ch/pub/juergen/nn_2005.pdf)__. In _Neural Networks_,  18/2005 (5-6): 602-10, Jun-Jul 2005.
- __[Greff et al.2015]__ Klaus Greff, Rupesh K. Srivastava, Jan Koutník, Bas R. Steunebrink, and Jürgen Schmidhuber. __[LSTM: A Search Space Odyssey (2015)](https://arxiv.org/abs/1503.04069)__. arXiv:1503.04069 [cs.NE], 2015.
- __[Hochreiter and Schmidhuber1997]__ Sepp Hochreiter and Jürgen Schmidhuber. __[Long short-term memory (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf)__. In _Neural Computation_, 9(8): 1735-1780, Nov. 1997.
- __[Joulin2016]__ Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. __[Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)__. arXiv:1607.01759 [cs.CL], 2016.
- __[Pascanu et al.2012]__ Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. __[On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063)__. arXiv:1211.5063 [cs.LG], 2012.
- __[Tang et al.2015]__ Duyu Tang, Bing Qin, and Ting Liu. __[Document Modeling with Gated Recurrent Neural Network for Sentiment Classification](http://aclweb.org/anthology/D15-1167)__. In _EMNLP_, 2015.
- __[Xiao and Cho2016]__ Yijun Xiao and Kyunghyun Cho. __[Efficient Character-level Document Classification by Combining Convolution and Recurrent Layers](https://arxiv.org/abs/1602.00367)__. arXiv:1602.00367 [cs.CL], 2016.
- __[Yogatama et al.2017]__ Dani Yogatama, Chris Dyer, Wang Ling, and Phil Blunsom.
__[Generative and Discriminative Text Classification with Recurrent Neural Networks](https://arxiv.org/abs/1703.01898)__. arXiv:1703.01898, 2017.
- __[Zhang and LeCun2015]__ Xiang Zhang and Yann LeCun. __[Text Understanding from Scratch](https://arxiv.org/abs/1502.01710)__. arXiv:1502.01710 [cs.LG], 2015.
- __[Zhang et al.2015]__ Xiang Zhang, Junbo Zhao, and Yann LeCun. __[Character-level convolutional networks for text classification](https://arxiv.org/abs/1509.01626)__. In _NIPS_: 649-657, 2015.
- __[Zhou et al.2015]__ Chunting Zhou, Chonglin Sun, Zhiyuan Liu, and Francis C.M. Lau. __[A C-LSTM Neural Network for Text Classification](https://arxiv.org/abs/1511.08630)__. arXiv:1511.08630 [cs.CL], 2015.
- __[Zhou et al.2016]__ Peng Zhou, Zhenyu Qi, Suncong Zheng, Jiaming Xu, Hongyun Bao, and Bo Xu. __[Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling](https://arxiv.org/abs/1611.06639)__. arXiv:1611.06639 [cs.CL], 2016.
- ...

### Books:
- __[Chollet2017]__ François Chollet. __[Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)__. Manning, 2017.
- ...

### Websites:
- __[Karpathy2015]__ Andrej Karpathy. __[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)__. May 2015.
- __[Olah2014]__ Christopher Olah. __[Understanding Convolutions](http://colah.github.io/posts/2014-07-Understanding-Convolutions/)__. July 2014.
- __[Olah2015a]__ Christopher Olah. __[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)__. August 2015.
- __[Olah2015b]__ Christopher Olah. __[Visual Information Theory](http://colah.github.io/posts/2015-09-Visual-Information/)__. October 2015.
- ...

### Documentations/APIs:
- [Keras: The Python Deep Learning library](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/get_started/), [API (Python)](https://www.tensorflow.org/api_docs/python/)
- [Cloud ML Engine Documentation](https://cloud.google.com/ml-engine/docs/)
- [scikit-learn](http://scikit-learn.org/stable/documentation.html)
- [Scrapy](https://docs.scrapy.org/en/latest/), [Scrapinghub](https://doc.scrapinghub.com/)
- [pandas](http://pandas.pydata.org/pandas-docs/stable/)
- [NumPy](https://docs.scipy.org/doc/numpy-1.13.0/reference/)
- [Python 3.6](https://docs.python.org/3.6/index.html), [2.7](https://docs.python.org/2/index.html)


---
##### Admin:
- [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
- [Markdown TOC in Atom](https://atom.io/packages/markdown-toc)
