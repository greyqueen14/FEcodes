import os
import pandas as pd
import nltk as nltk
import gensim
from gensim import corpora, models, similarities

nltk.download("punkt")
nltk.download("stopwords")

os.chdir("C:\\Users\\Dell\\Documents\\Python Scripts");
df=pd.read_csv('FinalData.csv');



x=df['Tweets'].values.tolist()
y=df['Authors'].values.tolist()


corpus= x+y
  
tok_corp= [nltk.word_tokenize(sent) for sent in corpus]


model = gensim.models.Word2Vec(tok_corp, min_count=1, size = 280)


#model.save('testmodel')
#model = gensim.models.Word2Vec.load('test_model')
#model.most_similar('word')
#model.most_similar([Vector])