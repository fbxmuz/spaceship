import spacy
import os

here = os.getcwd()
text = "terrible"
# Loading the best model from output_updated folder
# spacy load
nlp = spacy.load(os.path.join(here,"output_updated/model-best"))

demo = nlp(text)
print(demo.cats)