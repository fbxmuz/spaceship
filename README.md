# spaceship

My fist spacy project.

The code came from Dhilip Subramania's post in [Building a Sentiment Classifier using spaCy 3.0 Transformers](https://towardsdatascience.com/building-sentiment-classifier-using-spacy-3-0-transformers-c744bfc767b)

## Important Spacy commands on Windows

`pip install spacy=3.1.1 --user`

`pip install spacy-transformers --user`

`python -m spacy download encore_web_trf`

`python -m spacy init fill-config base_config.cfg > config.cfg`

`python -m spacy train config.cfg --verbose --output > output_updated`