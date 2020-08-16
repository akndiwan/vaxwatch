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
import requests
import numpy as np
from bs4 import BeautifulSoup

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
        temptime="Today, "+str(math.floor(((now - pubtime).seconds)/3600))+" hour(s) ago"
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


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

newsdf['clean_titles'] = newsdf['clean_titles'].apply(lemmatize_text)

def listToString(s):  
    
    # initialize an empty string 
    str1 = " " 
    
    # return string   
    return (str1.join(s)) 
        

newsdf['clean_titles'] = newsdf['clean_titles'].apply(listToString)



newStopWords = ["university","times","reuters","inc","international","organization","health","house","world","buy","pharma",'covid','sciences','hindustan','guide','institute','state','bbc','program','pharmaceuticals','innovations','cnn','grants','epidemic','respiratory','paycheck','protection','vaccine',"covid","coronavirus","today","trial","vaccines","could","trials","next","human","cancer","race","people","month","early","want","risk","science","pandemic","testing","second","says","start","phase","made","make","clinical","virus","update","update","final","treatment","research","nearly","begin","help","look","provides","kids","supply","stock","expert","doses","announces","plan","drug","new","hundreds","ceo","end","expected","millions","year","available","data","results","protect","may","months","scientists","researchers","billion","day", "report","study","need","development","complete","completes","heres","million","test","drugs","drug","potential","speed","boost","top","know","large","question","candidate","patient","talks",'patients',"use","i","ii","iii","track","claim","show","say","produce"]
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

r1 = requests.get("https://www.nytimes.com/interactive/2020/science/coronavirus-vaccine-tracker.html")
coverpage = r1.content
soup1 = BeautifulSoup(coverpage,'html.parser')

vaccines = soup1.find_all(class_='g-list-item')


org=[]
txt=[]
phases=[]

vacc_track=pd.DataFrame(columns=['org','phase','text'])

for i in range(0,len(vaccines)):
    rx=vaccines[i]
    rx=rx.text
    rx=rx.replace('\n',' ')
    data=[b.string for b in vaccines[i].findAll('b')]
    data=' & '.join(', '.join(data).rsplit(', ', 1))
    org.append(data)
    phs=(vaccines[i].select("span[class*=phase]"))
    temp2=""
    for i in range(0,len(phs)):
        temp=phs[i].text
        temp2=temp2+" "+temp
    phases.append(temp2)
    rx=rx.replace(temp2,'')
    txt.append(rx)

vacc_track.org=org
vacc_track.phase=phases
vacc_track.text=txt

preclinvacc = vacc_track[vacc_track.phase.str.contains("PRECLINICAL")].reset_index(drop=True)
phase3vacc = vacc_track[vacc_track.phase.str.contains("PHASE III")].reset_index(drop=True)
minusphase3= vacc_track[-(vacc_track.phase.str.contains("PHASE III"))].reset_index(drop=True)
phase2vacc = minusphase3[minusphase3.phase.str.contains("PHASE II")].reset_index(drop=True)
minusphase2= minusphase3[-(minusphase3.phase.str.contains("PHASE II"))].reset_index(drop=True)
phase1vacc = minusphase2[minusphase2.phase.str.contains("PHASE I")].reset_index(drop=True)



phase2vacc=json.loads(phase2vacc.to_json(orient='records'))
phase1vacc=json.loads(phase1vacc.to_json(orient='records'))
phase3vacc=json.loads(phase3vacc.to_json(orient='records'))
preclinvacc=json.loads(preclinvacc.to_json(orient='records'))


@app.route("/")
def home():
    return render_template('index.html', topdata1=topdata1, topdata2=topdata2, topdata3=topdata3, topdata4=topdata4, topdata510=topdata510, prtime =timestr,ph1vacc=phase1vacc, ph2vacc=phase2vacc,ph3vacc=phase3vacc,pcvacc=preclinvacc)


