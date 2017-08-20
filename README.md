# Deceptive Spam Review Detection with Convolutional Neural Network in Tensorflow

[**This code belongs to the "DECEPTIVE SPAM REVIEW DETECTION WITH CNN USING TENSORFLOW" blog post.**](https://migsena.com/deceptive_spam_part_1/)

we implement a model similar to the SCNN model of **Luyang Li's** [Document representation and feature combination for deceptive spam review detection](http://www.sciencedirect.com/science/article/pii/S0925231217303983). In that paper, the SCNN model apply convolutional neural network (CNN) technique to detect deceptive spam review. 

A Deceptive opinion spam is a review with fictitious opinions which is deliberately written to sound authentic. Deceptive spam review detection can then be thought as of the exercise of taking a review and determining whether is a spam or a truth.

## Requirements
* Python 2.7
* Jupyter Notebook
* Tensorflow
* Numpy

## Dataset
We will use [the first publicly available gold standard corpus of deceptive opinion spam](http://myleott.com/op_spam/). The dataset consists of truthful and deceptive hotel reviews of 20 Chicago hotels. It contains: 

* 400 truthful positive reviews from TripAdvisor
* 400 deceptive positive reviews from Mechanical Turk
* 400 truthful negative reviews from Expedia, Hotels.com, Orbitz, Priceline, TripAdvisor and Yelp
* 400 deceptive negative reviews from Mechanical Turk.

## References
[Document representation and feature combination for deceptive spam review detection](http://www.sciencedirect.com/science/article/pii/S0925231217303983)

[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

[Perform sentiment analysis with LSTMs, using TensorFlow](https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow)