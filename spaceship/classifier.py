import pandas as pd
from datetime import datetime
import spacy
import spacy_transformers

# Storing docs in binary format
from spacy.tokens import DocBin


n = [["positive", "yes"],
     ["negative", "no"],
     ["neutral", "maybe"],
     ["positive", "great"],
     ["negative", "terrible"],
     ["neutral", "so-so"],
     ["positive", "terrific"],
     ["negative", "horrible"],
     ["neutral", "ok"],
     ]

df = pd.DataFrame(n, columns=["Sentiment", 'Text'])

print(df.head())

train = df.sample(frac = 0.8, random_state = 25)
test = df.drop(train.index)

print(train.shape, test.shape)

import spacy
nlp=spacy.load("en_core_web_trf")


# Importing libraries

train['tuples'] = train.apply(lambda row:
(row['Text'],row['Sentiment']), axis=1)
train = train['tuples'].tolist()
test['tuples'] = test.apply(lambda row:
(row['Text'],row['Sentiment']), axis=1)
test = test['tuples'].tolist()


# User function for converting the train and test dataset into spaCy document
def document(data):
#Creating empty list called "text"
  text = []
  for doc, label in nlp.pipe(data, as_tuples = True):
    if (label=='positive'):
      doc.cats['positive'] = 1
      doc.cats['negative'] = 0
      doc.cats['neutral']  = 0
    elif (label=='negative'):
      doc.cats['positive'] = 0
      doc.cats['negative'] = 1
      doc.cats['neutral']  = 0
    else:
      doc.cats['positive'] = 0
      doc.cats['negative'] = 0
      doc.cats['neutral']  = 1
#Adding the doc into the list 'text'
      text.append(doc)
  return(text)


# Calculate the time for converting into binary document for train dataset

start_time = datetime.now()

#passing the train dataset into function 'document'
train_docs = document(train)

#Creating binary document using DocBin function in spaCy
doc_bin = DocBin(docs = train_docs)

#Saving the binary document as train.spacy
doc_bin.to_disk("train.spacy")
end_time = datetime.now()

#Printing the time duration for train dataset
print('Duration: {}'.format(end_time - start_time))

# Calculate the time for converting into binary document for test dataset

start_time = datetime.now()

#passing the test dataset into function 'document'
test_docs = document(test)
doc_bin = DocBin(docs = test_docs)
doc_bin.to_disk("valid.spacy")
end_time = datetime.now()

#Printing the time duration for test dataset
print('Duration: {}'.format(end_time - start_time))


# Calculate the time for converting into binary document for test dataset

start_time = datetime.now()

#passing the test dataset into function 'document'
test_docs = document(test)
doc_bin = DocBin(docs = test_docs)
doc_bin.to_disk("test.spacy")
end_time = datetime.now()

#Printing the time duration for test dataset

print('Duration: {}'.format(end_time - start_time))

