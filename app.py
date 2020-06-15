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
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import locale
import numpy as np
from PIL import Image
from nltk.tag import StanfordNERTagger
import os
import nltk
from collections import Counter
import spacy
import en_core_web_sm
import numpy as np
from PIL import Image
import nltk

from wordcloud import ImageColorGenerator
import matplotlib.pyplot as plt
from wordcloud import WordCloud

app = Flask(__name__)

API_KEY="ecbb55b0f6a941a2acfe909ee880eb55"
topic="covid vaccine trial"
yestdate=(datetime.today() - timedelta(2)).strftime('%Y-%m-%d')

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
        text = text.lower()
        text = re.sub('<[^>]*>', ' ', text)
        #text = re.sub('[\W]+', ' ', text)
        text = re.sub('xa0', '', text)
        text = text.lower()
        text= re.sub("'", "", text)
        text= re.sub(",", "", text)
        text = re.sub(r"\d", "", text)
        text = re.sub(r"[,@\'?\.$%_]", "", text, flags=re.I)
        text = text.replace("''", "")
        text = text.replace('"', "")
        text = text.replace('coronavirus', "")
        text = text.replace('june', "")
        text = text.replace('covid', "")
        text = text.replace('daily', "")
        text = text.replace('news', "")
        text = text.replace('july', "")
        text = text.replace('first', "")
        text = text.replace('second', "")
        return text

newsdf['clean_titles']=newsdf.title.apply(preprocessor)
stop_words = nltk.corpus.stopwords.words('english')


newStopWords = ["university","times","reuters","inc","international","organization","health","house","world","buy","pharma",'covid','sciences','hindustan','guide','institute','state','bbc','program','pharmaceuticals','innovations','cnn','grants','epidemic','respiratory','paycheck','protection','vaccine',"covid","coronavirus","today","trial","vaccines","could","trials","next","human","cancer","race","people","month","early","want","risk","science","pandemic","testing","second","says","start","phase","made","make","clinical","virus","update","update","final","treatment","research","nearly","begin","help","look","provides","kids","supply","stock","expert","doses","announces","plan","drug"]
_mask = np.array(Image.open("static/images/wordcloud.jpg").convert('RGB'))
_mask[_mask==0]=255
stop_words.extend(newStopWords)

wordcloud2 = WordCloud(background_color="white",width = 200,height = 200, collocations=False,stopwords = stop_words, max_words = 500,min_font_size = 3, mask=_mask, random_state=42).generate(' '.join(newsdf['clean_titles']))
fig = plt.figure(figsize = (20,10))
image_colors = ImageColorGenerator(_mask)
plt.imshow(wordcloud2.recolor(color_func=image_colors), interpolation = "bilinear")
plt.axis('off')
plt.tight_layout(pad=0)

#plt.show()
plt.savefig("static/images/VaxWC.png",format="png",bbox_inches='tight',pad_inches = 0)
#plt.show()



@app.route("/")
def home():
    return render_template('index.html', topdata1=topdata1, topdata2=topdata2, topdata3=topdata3, topdata4=topdata4, topdata510=topdata510, prtime =timestr )


