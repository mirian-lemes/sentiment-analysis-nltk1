

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *


nltk.download('subjectivity')


n_instances = 100


subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]


train_docs = subj_docs + obj_docs


sentim_analyzer = SentimentAnalyzer()


all_words = sentim_analyzer.all_words([doc for doc in train_docs])


unigram_feats = sentim_analyzer.unigram_word_feats(all_words, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)


training_set = sentim_analyzer.apply_features(train_docs)


trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)


testes = [
    "This car is beautiful",
    "This car is horrible",
    "I like this perfume",
    "I hate this perfume",
    "I love it",
    "This series is terrible",
    "This series is amazing",
    "Great, the bread is gone",
    "This sneaker is not good"
]


print("Resultados dos testes:\n")
for frase in testes:
    tokens = frase.split()
    features = extract_unigram_feats(tokens, unigrams=unigram_feats)
    resultado = classifier.classify(features)
    print(frase, "->", resultado)