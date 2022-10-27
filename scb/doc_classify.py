import pandas as pd
import os
import datetime
import torch
import unidecode 
import re
import time 
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from autocorrect import Speller 
from bs4 import BeautifulSoup 
from nltk.corpus import stopwords 
from nltk import word_tokenize
import string
import shutil
from scipy.special import softmax
from pyparsing import results

from transformers import BertForSequenceClassification, BertTokenizer, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from cdqa.utils.converters import pdf_converter

nltk.download('stopwords') 
nltk.download('punkt')
nltk.download('wordnet')

model = BertForSequenceClassification.from_pretrained(r'content/model_save')
tokenizer = BertTokenizer.from_pretrained(r'content/model_save',do_lower_case=True)
MAX_LEN = 124

if torch.cuda.is_available():
  device = torch.device("cuda")
  print("Gpu count", torch.cuda.device_count())
  print("GPU name",torch.cuda.get_device_name(0))
else:
  device = torch.device("cpu")

stoplist = stopwords.words('english') 
stoplist = set(stoplist)
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def format_time(elapsed):
  elapsed_rounded = int(round(elapsed))
  return str(datetime.timedelta(seconds=elapsed_rounded))

def remove_newlines_tabs(text):
    Formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
    return Formatted_text

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_links(text):
    remove_https = re.sub(r'http\S+', '', text)
    remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
    return remove_com

def remove_whitespace(text):
    pattern = re.compile(r'\s+') 
    Without_whitespace = re.sub(pattern, ' ', text)
    text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
    return text

def accented_characters_removal(text): 
    text = unidecode.unidecode(text)
    return text

def lower_casing_text(text):
    text = text.lower()
    return text

def reducing_incorrect_character_repeatation(text):
    # Pattern matching for all case alphabets
    Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)
    
    # Limiting all the  repeatation to two characters.
    Formatted_text = Pattern_alpha.sub(r"\1\1", text) 
    
    # Pattern matching for all the punctuations that can occur
    Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')
    
    # Limiting punctuations in previously formatted string to only one.
    Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)
    
    # The below statement is replacing repeatation of spaces that occur more than two times with that of one occurrence.
    Final_Formatted = re.sub(' {2,}',' ', Combined_Formatted)
    return Final_Formatted

def removing_special_characters(text):
    Formatted_Text = re.sub(r"[^a-zA-Z0-9:$-,%.?!]+", ' ', text)
    return Formatted_Text

def removing_stopwords(text):
    text = repr(text)
    No_StopWords = [word for word in word_tokenize(text) if word.lower() not in stoplist ]
    words_string = ' '.join(No_StopWords)    
    return words_string

def spelling_correction(text):
    spell = Speller(lang='en')
    Corrected_text = spell(text)
    return Corrected_text

def lemmatization(text):
    lemma = [lemmatizer.lemmatize(w,'v') for w in w_tokenizer.tokenize(text)]
    return lemma

def test_data_pre(docs):
  test_input_ids = []
  for sen in docs.text:
    if ((len(test_input_ids) % 20000) == 0):
      print(len(test_input_ids))
    encoded_sent = tokenizer.encode(
        sen,
        add_special_tokens = True,
        max_length = MAX_LEN,
    )
    test_input_ids.append(encoded_sent)
  print(len(test_input_ids))

  #test_labels = docs['labels'].values
  test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long",value=0,truncating="post",padding="post")
  test_attention_masks = []
  for seq in test_input_ids:
    seq_mask = [float(i > 0) for i in seq]
    test_attention_masks.append(seq_mask)

  test_inputs = torch.tensor(test_input_ids)
  test_mask = torch.tensor(test_attention_masks)
  #test_labels = torch.tensor(test_labels)
  batch_size = 32
  test_data = TensorDataset(test_inputs, test_mask)
  test_sampler = RandomSampler(test_data)
  test_dataloder = DataLoader(test_data, sampler=test_sampler, batch_size = batch_size)
  model.eval()
  predictions, true_labels = [],[]
  t0 = time.time()
  for(step, batch) in enumerate(test_dataloder):
    print("test")
    batch = tuple(t.to(device) for t in batch)
    if step % 100 == 0 and not step == 0:
      elapsed = format_time(time.time()-t0)

      print(' Batch {:>5,} of {:>5,}.   Elapsed: {:}.'.format(step, len(test_dataloder), elapsed))

    b_input_ids, b_input_mask = batch

    with torch.no_grad():
      outputs = model(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask)
      
    logits = outputs[0]
    print("logits",logits)
    logits = logits.detach().cpu().numpy()

    predictions.append(logits)

  return predictions

def pred_confidence(V1,V2): 
  dif = V1 - V2
  return dif
#   if dif > 0.3:
#     return "High Confidence"
#   else:
#     return "Low Confidence"


def class_predict(ippath):
    print("\n\n\n\n This is ippath \n\n\n\n\n",ippath)
    df_test = pdf_converter(directory_path=ippath)
    print("\n\n\n\nThis is df_test\n\n\n",df_test)
    df_test.columns = ['title','text']
    print(    print("\n\n\n\nThis is df_test[text]\n\n\n",df_test['text'][0])
)

    for i in range(df_test.shape[0]):
        vaal =''
        vaal = df_test.loc[i, 'text']
        df_test.loc[i, 'text'] = str(vaal)

    df_test['text'] = df_test['text'].apply(remove_newlines_tabs)
    df_test['text'] = df_test['text'].apply(strip_html_tags)
    df_test['text'] = df_test['text'].apply(remove_links)
    df_test['text'] = df_test['text'].apply(remove_whitespace)
    df_test['text'] = df_test['text'].apply(accented_characters_removal)
    df_test['text'] = df_test['text'].apply(lower_casing_text)
    df_test['text'] = df_test['text'].apply(reducing_incorrect_character_repeatation)
    df_test['text'] = df_test['text'].apply(removing_special_characters)
    df_test['text'] = df_test['text'].apply(removing_stopwords)
    #docs['paragraphs'] = docs['paragraphs'].apply(spelling_correction)
    df_test['text'] = df_test['text'].apply(lemmatization)

    for i in range(df_test.shape[0]):
        nlist = [',','.','',' ']
        cnt = ""
        for val in df_test.iloc[i,1]:
            if val not in nlist:
                cnt = cnt+' '+val
        df_test.iloc[i,1] = cnt

    predict_cls = []

    final_prediction = {}
    for i in range(df_test.shape[0]):
        print(i)
        df_c = pd.DataFrame({'title' : [df_test['title'][i]],'text' : df_test['text'][i]})
        pred = test_data_pre(df_c)
        pred1 = softmax(pred)
        final_prediction[df_test.iloc[i,0]] = pred1
    
    for k,v in final_prediction.items():
        result = v[0][0]
        #print(result)
        mx = max(result[0],result[1])
     
        if result[0] > result[1]:
            k = k+".pdf"
            print(k)
            cls1 = "Form 10-k"
            F10Kfpath = r"PocApp/static/pdf/"+k
            dstpath = r"PocApp/static/classify_docs/" + cls1
            os.makedirs(r"PocApp/static/classify_docs", exist_ok=True)
            os.makedirs(dstpath, exist_ok=True)
            shutil.copyfile(F10Kfpath, dstpath+'/'+k)
            print(cls1 + "File copied successfully.")
            print(k+":"+" Form-10K"+" with "+str(round(mx * 100)))
            predict_cls.append([k,cls1, str(round(mx * 100))])
        else:
            k = k+".pdf"
            print(k)
            cls2 = "LPA"
            LPAfpath = r"PocApp/static/pdf/"+k
            dstpath = r"PocApp/static/classify_docs/" + cls2
            os.makedirs(r"PocApp/static/classify_docs", exist_ok=True)
            os.makedirs(dstpath, exist_ok=True)
            print(cls2 + "File copied successfully.")
            shutil.copyfile(LPAfpath, dstpath+"/"+k)
            print(k+":"+" LPA"+" with "+str(round(mx * 100)))
            predict_cls.append([k,cls2, str(round(mx * 100))])

    print(predict_cls)
    df_cls = pd.DataFrame(predict_cls, columns = ['Document', 'Document_Type', 'Conf_Score'])
    return df_cls
