#!/usr/bin/env python
# coding: utf-8

# SPAM CLASSIFIER USING NAIVE BAYES THEOREM. 
# 

# We are using sklearn.naive_bayes to train a spam classifier. 

# Basically the code creates a dataframe having messages and class as columns and most of the code is about extracting the messages from the files.
# Then after extracting we have just applied naive bayes to it.

# In[1]:


import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})#creates a database which have 2 columns one is the messages and other whether spam or ham

data = data.append(dataFrameFromDirectory('/home/khushboopriya/spam', 'spam'))
data = data.append(dataFrameFromDirectory('/home/khushboopriya/ham', 'ham'))


# Let's have a look at that DataFrame:

# In[2]:


data.head()


# In[3]:


data[20:30]


# Now we will use a CountVectorizer to split up each message into its list of words, and throw that into a MultinomialNB classifier. Call fit() and we've got a trained spam filter ready to go! It's just that easy.

# In[6]:


vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)


# Let's try it out:

# In[4]:


examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
predictions


# In[ ]:




