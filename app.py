# Imports
import numpy as np
import pandas as pd
import torch
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizer,  AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, DetectorFactory
import nltk
import re

# Data Processing
def preprocess_data(data_path, size):
    data = pd.read_json(data_path, lines=True)
    data = data.dropna(subset=['abstract']).reset_index(drop=True)
    data = data.sample(size)[['abstract', 'id']]
    data.rename(columns={'id': 'paperid'}, inplace=True)
    return data

data_path = "./arxiv_cs.json"
source_data = preprocess_data(data_path, 10)


# Embedding Creation
model_path = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path, 
                                          do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                          output_attentions=False,
                                                          output_hidden_states=True)

def create_vector_from_text(tokenizer, model, text, MAX_LEN = 510):
    input_ids = tokenizer.encode(
                        text, 
                        add_special_tokens = True, 
                        max_length = MAX_LEN,                           
                   )    

    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long", 
                              truncating="post", padding="post")
    
    input_ids = results[0]
    attention_mask = [int(i>0) for i in input_ids]
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    model.eval()
    with torch.no_grad():        
        logits, encoded_layers = model(
                                    input_ids = input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = attention_mask,
                                    return_dict=False)

    layer_i = 12 
    batch_i = 0 
    token_i = 0 
        
    vector = encoded_layers[layer_i][batch_i][token_i]
    vector = vector.detach().cpu().numpy()
    return(vector)

def create_vec_db(data):
    vectors = []
    source_data = data.abstract.values
    for text in source_data:        
        vector = create_vector_from_text(tokenizer, model, text)
        vectors.append(vector)
    
    data["vectors"] = vectors
    data["vectors"] = data["vectors"].apply(lambda emb: np.array(emb))
    data["vectors"] = data["vectors"].apply(lambda emb: emb.reshape(1, -1))
    return data

stopwords = nltk.corpus.stopwords.words('english')
def clean_text(text):
    edited_text=re.sub('\W'," ",text) #replace any sumbol with whitespace
    edited_text=re.sub("  "," ",edited_text) #replace double whitespace with single whitespace
    edited_text=edited_text.split(" ") #split the sentence into array of strings
    edited_text=" ".join([char for char in edited_text if char!= ""]) #remove any empty string from text
    edited_text=edited_text.lower() #lowercase
    edited_text=re.sub('\d+',"",edited_text) #Removing numerics
    edited_text=re.split('\W+',edited_text) #spliting based on whitespace or whitespaces
    edited_text=" ".join([word for word in edited_text if word not in stopwords])
    return edited_text

vec_db = create_vec_db(source_data)
vec_db['abstract']=vec_db.abstract.apply(lambda x: clean_text(x))


# Translation, Paraphrase & Similarity


language_list = ['de', 'fr', 'el', 'ja', 'ru']
DetectorFactory.seed = 0

def translate_text(text, text_lang, target_lang='en'):
  model_name = f"Helsinki-NLP/opus-mt-{text_lang}-{target_lang}"
  tokenizer = MarianTokenizer.from_pretrained(model_name)
  model = MarianMTModel.from_pretrained(model_name)
  formated_text = ">>{}<< {}".format(text_lang, text)
  translation = model.generate(**tokenizer([formated_text], return_tensors="pt", padding=True))
  translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translation][0]
  return translated_text

def process_document(text):
    text_vect = create_vector_from_text(tokenizer, model, clean_text(text))
    text_vect = np.array(text_vect)
    text_vect = text_vect.reshape(1, -1)
    return text_vect

    
def is_plagiarism(similarity_score, plagiarism_threshold):
  is_plagiarism = False
  if(similarity_score >= plagiarism_threshold):
    is_plagiarism = True
  return is_plagiarism

# 

def check_incoming_document(incoming_document):
  text_lang = detect(incoming_document)
  # Supported languages - French, German, Greek, Japanese, Russian
  language_list = ['de', 'fr', 'el', 'ja', 'ru']
  final_result = ""
  if(text_lang == 'en'):
    final_result = incoming_document 
  elif(text_lang not in language_list):
    final_result = None
  else:
    final_result = translate_text(incoming_document, text_lang)
  return final_result

def generate_paraphrases(para):
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to('cpu')
    sentence = "Here are two questions (Question1 and Question2). If these questions have the same meaning and same answer, answer 'Yes', otherwise 'No'.\nQuestion1: Would the idea of Trump and Putin in bed together scare you, given the geopolitical implications?, Question2: Do you think that if Donald Trump were elected President, he would be able to restore relations with Putin and Russia as he said he could, based on the rocky relationship Putin had with Obama and Bush?\n"

    text =  "paraphrase: " + para + " </s>"

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to("cpu"), encoding["attention_mask"].to("cpu")

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=3
    )
    h=[]
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        print(line)
        h.append(line)
    return h
    

    
def run_plagiarism_analysis(query_text, data, plagiarism_threshold=0.8):

    top_N=3
    document_translation = check_incoming_document(query_text)

    paraphrased_texts = generate_paraphrases(document_translation)

    # document_translation = document_translation['original']
    res = {}
    if(document_translation is not None):
      # Generate paraphrases of the incoming text
      
      query_vect = process_document(document_translation)
      
      # Run similarity Search
      data["similarity"] = data["vectors"].apply(lambda x: cosine_similarity(query_vect, x))
      data["similarity"] = data["similarity"].apply(lambda x: x[0][0])

      similar_articles = data.sort_values(by='similarity', ascending=False)[0:top_N+1]
      formated_result = similar_articles[["abstract", "paperid", "similarity"]].reset_index(drop = True)

      similarity_score = formated_result.iloc[0]["similarity"] 
      most_similar_article = formated_result.iloc[0]["abstract"] 
      is_plagiarism_bool = is_plagiarism(similarity_score, plagiarism_threshold)

      plagiarism_decision = {'similarity_score': similarity_score, 
                             'is_plagiarism': is_plagiarism_bool,
                             'most_similar_article': most_similar_article, 
                             'article_submitted': query_text
                            }
      res['original'] = plagiarism_decision
    
    paraphrased_detect = []
    for paraphrased_text in paraphrased_texts:
      query_vect = process_document(paraphrased_text)
      
      # Run similarity Search
      data["similarity"] = data["vectors"].apply(lambda x: cosine_similarity(query_vect, x))
      data["similarity"] = data["similarity"].apply(lambda x: x[0][0])

      similar_articles = data.sort_values(by='similarity', ascending=False)[0:top_N+1]
      formated_result = similar_articles[["abstract", "paperid", "similarity"]].reset_index(drop = True)

      similarity_score = formated_result.iloc[0]["similarity"] 
      most_similar_article = formated_result.iloc[0]["abstract"] 
      is_plagiarism_bool = is_plagiarism(similarity_score, plagiarism_threshold)

      plagiarism_decision = {'similarity_score': similarity_score, 
                            'is_plagiarism': is_plagiarism_bool,
                            'paraphrased_text': paraphrased_text,
                            'most_similar_article': most_similar_article, 
                            'article_submitted': query_text
                            } 
      paraphrased_detect.append(plagiarism_decision)
    res['paraphrased'] = paraphrased_detect
    return res

# Streamlit App
import streamlit as st
import pypdf
from io import BytesIO

def extract_text_from_pdf(pdf_file):
    pages = []
    pdf = pypdf.PdfReader(pdf_file)
    for p in range(len(pdf.pages)//2):
      page = pdf.pages[p]
      text = page.extract_text()
      page2 = pdf.pages[p+1]
      text2 = page2.extract_text() 
      pages.append(text+text2)
    return pages

def main():
    st.title("AI Research Paper Plagiarism Checker")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)

        st.write("Extracted Text:")
        st.text(pdf_text[:500])  

        # Plagiarism analysis
        if st.button("Check for Plagiarism"):
          for page in pdf_text:
            analysis_result = run_plagiarism_analysis(page, vec_db, plagiarism_threshold=0.8)
            st.write(analysis_result)
            st.write("----------------------------------------------------")

if __name__ == "__main__":
    main()
    # french_article_to_check = """
    # Les Réseaux d’Innovation et de Transfert Agricole (RITA) ont été créés en 2011 pour mieux connecter la recherche et le développement agricole, intra et inter-DOM, avec un objectif d’accompagnement de la diversification des productions locales. Le CGAAER a été chargé d'analyser ce dispositif et de proposer des pistes d'action pour améliorer la chaine Recherche – Formation – Innovation – Développement – Transfert dans les outre-mer dans un contexte d'agriculture durable, au profit de l'accroissement de l'autonomie alimentaire.
    # """
    # analysis_result = run_plagiarism_analysis(french_article_to_check, vec_db, plagiarism_threshold=0.8)
    # analysis_result