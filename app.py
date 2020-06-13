from flask import Flask, render_template
from datetime import datetime, timedelta
import re
import json
import requests
import pandas as pd
import requests
import math 
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import locale
import numpy as np
from PIL import Image
from nltk.tag import StanfordNERTagger
import os
import nltk


app = Flask(__name__)

API_KEY="ecbb55b0f6a941a2acfe909ee880eb55"
topic="covid vaccine trial"
yestdate=(datetime.today() - timedelta(1)).strftime('%Y-%m-%d')

url = ('http://newsapi.org/v2/everything?q='+topic+'&from='+yestdate+'&sortBy=publishedAt&apiKey='+API_KEY+'&language=en&pageSize=100&page=1&sortBy=popularity&language=en')
response = requests.get(url)
news=(response.json())

def flatten_json(nested_json, exclude=['']):
    out = {}

    def flatten(x, name='', exclude=exclude):
        if type(x) is dict:
            for a in x:
                if a not in exclude: flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out


def cleandesc(text):
    text = re.sub('<[^>]*>', ' ', text)
    return text



newsdf=pd.DataFrame([flatten_json(x) for x in news['articles']])

newsdf.fillna(value="NA", inplace=True)

newsdf = newsdf.drop_duplicates(subset=['title'], keep='first')
newsdf["VaccRelated"]= (newsdf.description.str.contains("vaccine") | newsdf.title.str.contains("vaccine") | newsdf.description.str.contains("Vaccine") | newsdf.title.str.contains("Vaccine"))

vaccnews=newsdf[newsdf.VaccRelated == 1]

vaccnews.description=vaccnews.description.apply(cleandesc)

now= datetime.strptime(datetime.utcnow().replace(microsecond=0).isoformat(), '%Y-%m-%dT%H:%M:%S')
timeSince=[]
for i in range(0,len(vaccnews)):
    pubtime =datetime.strptime(vaccnews.publishedAt.iloc[i], '%Y-%m-%dT%H:%M:%SZ')
    if((now-pubtime).days==0):
        temptime="Today, "+str(math.floor(((now - pubtime).seconds)/3600))+" hours ago"
    else:
        temptime=str((now-pubtime).days)+" day(s), "+str(math.floor(((now - pubtime).seconds)/3600))+" hours ago"
    timeSince.append(temptime)
vaccnews["timeSince"]=timeSince


top_vacc=vaccnews[0:1]
top_vacc2=vaccnews[2:3]
top_vacc3=vaccnews[3:4]
top_vacc4=vaccnews[4:5]
top_vacc510=vaccnews[5:10]

topdata1=json.loads(top_vacc.to_json(orient='records'))
topdata2=json.loads(top_vacc2.to_json(orient='records'))
topdata3=json.loads(top_vacc3.to_json(orient='records'))
topdata4=json.loads(top_vacc4.to_json(orient='records'))
topdata510=json.loads(top_vacc510.to_json(orient='records'))

now = datetime.utcnow()
timestr = str(now.strftime("%A, %d %B, %Y"))

text=str(newsdf["title"].values)

def preprocessor(text):
    if type(text) == str:
        text = re.sub('<[^>]*>', ' ', text)
        text = re.sub('[\W]+', ' ', text)
        text = re.sub('xa0', '', text)
        text=text.replace('Inc', '')
        text=text.replace('Inc.', '')
        text=text.replace('Times', '')
        text=text.replace('News', '')
        return text
    
text=preprocessor(text)

# set java path
java_path ='file:///Library/Java/JavaVirtualMachines/jdk-13.0.2.jdk/Contents/Home/'
os.environ['JAVAHOME'] = java_path
# initialize NER tagger
sn = StanfordNERTagger('/Users/akndiwan/StanfordNLP/stanford-ner-4.0.0/classifiers/english.all.3class.distsim.crf.ser.gz',
                       path_to_jar='/Users/akndiwan/StanfordNLP/stanford-ner-4.0.0/stanford-ner.jar')
# tag named entities
words = nltk.word_tokenize(text)
tags=(sn.tag(words))
# extract all named entities
named_entities = []
temp_named_entity=None
temp_entity_name = ''
for term, tag in tags:
    if tag != 'O':
        temp_entity_name = ' '.join([temp_entity_name, term]).strip()
        temp_named_entity = (temp_entity_name, tag)
    else:
        if temp_named_entity:
            named_entities.append(temp_named_entity)
            temp_entity_name = ''
            temp_named_entity = None
#named_entities = list(set(named_entities))
entity_frame = pd.DataFrame(named_entities, 
                            columns=['Entity Name', 'Entity Type'])
# view top entities and types
top_entities = (entity_frame.groupby(by=['Entity Name', 'Entity Type'])
                           .size()
                           .sort_values(ascending=False)
                           .reset_index().rename(columns={0 : 'Frequency'}))
top_entities = (entity_frame.groupby(by=['Entity Name', 'Entity Type'])
                           .size()
                           .sort_values(ascending=False)
                           .reset_index().rename(columns={0 : 'Frequency'}))
ImpEntities=top_entities
#[(top_entities["Entity Type"]=="ORGANIZATION") | (top_entities["Entity Type"]=="PERSON")]
text=str(ImpEntities["Entity Name"].values)
text=text.replace("'", "")
text=re.sub("(\\d|\\W)+"," ",text)

stop_words = nltk.corpus.stopwords.words('english')
stem_desc=[]
split_text = text.split()
#Lemmatisation
lem = WordNetLemmatizer()
split_text = [lem.lemmatize(word) for word in split_text if not word in stop_words] 
split_text = " ".join(split_text)
#stem_desc.append(split_text)
split_text=split_text.lower()
newStopWords = ["university","times","reuters","inc","international","associated","press","organization","health","house","hill","concerns","rapid,""world","buy","pharma",'covid','coronavirus','sciences','hindustan','guide','institute','new','york','state','bbc','program','pharmaceuticals','innovations','cnn','grants','epidemic','respiratory','paycheck','protection','cnet','finance','commerce','bank','healthcare','systems','vaccine','right','top','preventative','shows','college','stock','children','promise','power','yearly','shows','global','republic','research','fate','market','podcast','transcript','city',"serum","journal","clinic","candidate","antibody","start","initiative","plans","report","engineer","phase","genetic","business","stocks","response","cocktail","reopening"]
stop_words.extend(newStopWords)
_mask = np.array(Image.open("static/images/covidcloud.jpg").convert('RGB'))
_mask[_mask==0]=255
#set the word cloud parameters
wordcloud = WordCloud( background_color="white",width = 200,height =200, stopwords = stop_words, max_words = 100000,min_font_size = 5, mask=_mask, random_state=42,mode="RGBA", max_font_size=50).generate(split_text)
#plot the word cloud
fig = plt.figure(figsize = (20,10))
image_colors = ImageColorGenerator(_mask)
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation = "bilinear")
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig("static/images/VaxWC.png",format="png")
#plt.show()



@app.route("/")
def home():
    return render_template('index.html', topdata1=topdata1, topdata2=topdata2, topdata3=topdata3, topdata4=topdata4, topdata510=topdata510, prtime =timestr )

