#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Hamza Farooq
"""

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import os
nlp = spacy.load("en_core_web_sm")
from spacy import displacy
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import nltk
nltk.download('stopwords')
import geonamescache

import os
import streamlit as st
import utils as utl
from PIL import Image
import time
import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration
tr = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
mdl = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
torch_device = 'gpu'


def main():
    # Settings
    st.set_page_config(layout="wide", page_title='New York Hotels')
    def bart_summarize(text, num_beams=20, length_penalty=2, max_length=2048, min_length=56, no_repeat_ngram_size=2):

      text = text.replace('\n','')
      text_input_ids = tr.batch_encode_plus([text], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)
      summary_ids = mdl.generate(text_input_ids, num_beams=int(num_beams), length_penalty=float(length_penalty), max_length=int(max_length), min_length=int(min_length), no_repeat_ngram_size=int(no_repeat_ngram_size))
      summary_txt = tr.decode(summary_ids.squeeze(), skip_special_tokens=True)
      return summary_txt


    gc = geonamescache.GeonamesCache()

    # gets nested dictionary for countries
    countries = gc.get_countries()

    # gets nested dictionary for cities
    cities = gc.get_cities()
    # def gen_dict_extract(var, key):
    #     if isinstance(var, dict):
    #         for k, v in var.items():
    #             if k == key:
    #                 yield v
    #             if isinstance(v, (dict, list)):
    #                 yield from gen_dict_extract(v, key)
    #     elif isinstance(var, list):
    #         for d in var:
    #             yield from gen_dict_extract(d, key)
    #
    # cities = [*gen_dict_extract(cities, 'name')]
    # countries = [*gen_dict_extract(countries, 'name')]
    #
    # cities.append('New York')

    from nltk.corpus import stopwords

    stopwords = set(stopwords.words('english'))
    #mask = np.array(Image.open('upvote.png'))

    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    import matplotlib.pyplot as plt
    #original_title = '<p style="font-family:IBM Mono; color:Blue; font-size: 20px;">Original image</p>'
    st.title("New York Hotel Finder")


    stopwords=list(STOP_WORDS)
    stopwords.extend(['hotel','room','rooms'])
    from string import punctuation
    punctuation=punctuation+ '\n'

    import pandas as pd
    from sentence_transformers import SentenceTransformer
    import scipy.spatial
    import pickle as pkl
    from sentence_transformers import SentenceTransformer, util
    import torch
    #import os

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    df_all = pd.read_csv('Hotel New York Combined.csv')

    df_all = df_all[['hotel_name','review_body']]
    #
    # df['hotel_name'].drop_duplicates()

    # df_combined = df.sort_values(['hotel_name']).groupby('hotel_name', sort=False).review_body.apply(''.join).reset_index(name='all_review')

    import re

    df_combined = pd.read_csv('df_combined.csv')

    # df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
    #
    # def lower_case(input_str):
    #     input_str = input_str.lower()
    #     return input_str
    #
    # df_combined['all_review']= df_combined['all_review'].apply(lambda x: lower_case(x))

    df = df_combined

    df_sentences = df_combined.set_index("all_review")

    df_sentences = df_sentences["hotel_name"].to_dict()
    df_sentences_list = list(df_sentences.keys())

    import pandas as pd
    from tqdm import tqdm
    from sentence_transformers import SentenceTransformer, util

    df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]
    #
    corpus = df_sentences_list
    corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)
    #
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # paraphrases = util.paraphrase_mining(model, corpus)

    #queries = ['Hotel close to Central Park',
    #           'Hotel with breakfast'
    #           ]


    # from transformers import AutoTokenizer, AutoModel
    # import torch
    # import torch.nn.functional as F
    #
    # #Mean Pooling - Take attention mask into account for correct averaging
    # def mean_pooling(model_output, attention_mask):
    #     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    #
    #
    # # Sentences we want sentence embeddings for
    # sentences = corpus
    #
    # # Load model from HuggingFace Hub
    # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v1')
    # model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v1')
    #
    # # Tokenize sentences
    # encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    #
    # # Compute token embeddings
    # with torch.no_grad():
    #     model_output = model(**encoded_input)
    #
    # # Perform pooling
    # sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    #
    # # Normalize embeddings
    # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    #
    # st.text("Sentence embeddings:")
    # st.text(sentence_embeddings)
    #
    #

    #corpus_embeddings = sentence_embeddings
    # Query sentences

    def plot_cloud(wordcloud):
        # Set figure size
        st.pyplot.figure(figsize=(40, 30))
        # Display image
        st.pyplot(wordcloud)
        # No axis details
        #st.pyplot.axis("off");
    userinput = st.text_input('Tell us what are you looking in your hotel?')
    if not userinput:
        st.write("Please enter a query to get results")
    else:
        query = [str(userinput)]
        doc = nlp(str(userinput))
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                if ent.text in countries:
                    st.write(f"Country : {ent.text}")
                elif ent.text in cities:
                    st.write("city")
                    st.write(ent.text)
                    st.write(f"City : {ent.text}")
                else:
                    print(f"Other GPE : {ent.text}")
        # query_embeddings = embedder.encode(queries,show_progress_bar=True)
        top_k = min(5, len(corpus))

        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        # st.write("\n\n======================\n\n")
        # st.write("Query:", query)
        # # doc = nlp(query)
        sentence_spans = list(doc.sents)
        ent_html = displacy.render(doc, style="ent", jupyter=False)
# Display the entity visualization in the browser:
        st.markdown(ent_html, unsafe_allow_html=True)

        #displacy.render(doc, jupyter = True, style="ent")
        st.write("##")
        st.subheader("\n\n\n\n\n\nTop 5 most relevant hotels:\n\n\n\n\n\n\n")
        st.write("\n\n======================\n\n")

        for score, idx in zip(top_results[0], top_results[1]):

            row_dict = df.loc[df['all_review']== corpus[idx]]
            st.subheader(row_dict['hotel_name'].values[0])
            hotel_subset = df_all.loc[df_all['hotel_name']==row_dict['hotel_name'].values[0]]
            st.caption("Review Summary:")
            st.write(row_dict['summary'].values[0])
            st.caption("Relevancy: {:.4f}".format(score))
            st.caption("Relevant reviews:")

            df_sentences_h = hotel_subset.set_index("review_body")

            df_sentences_h = df_sentences_h["hotel_name"].to_dict()
            df_sentences_list_h = list(df_sentences_h.keys())



            df_sentences_list_h = [str(d) for d in tqdm(df_sentences_list_h)]
            #
            corpus_h = df_sentences_list_h
            corpus_embeddings_h = embedder.encode(corpus_h,show_progress_bar=True)
            cos_scores_h = util.pytorch_cos_sim(query_embedding, corpus_embeddings_h)[0]
            top_results_h = torch.topk(cos_scores_h, k=top_k)

            for score, idx in zip(top_results_h[0], top_results_h[1]):
                st.write(corpus_h[idx])

            # st.table(hotel_subset.head())

            # st.write("#")
            #wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='navy', colormap='rainbow', collocations=False, stopwords = STOPWORDS, mask=mask).generate(corpus[idx])
            # wordcloud = WordCloud(collocations=False,stopwords=stopwords,background_color='black',max_words=35).generate(corpus[idx])
            # fig, ax = plt.subplots()
            # plt.imshow(wordcloud, interpolation='bilinear')
            # plt.axis("off")
            # plt.show()
            # st.pyplot(fig)
            # st.set_option('deprecation.showPyplotGlobalUse', False)


if __name__ == '__main__':
    main()


    # cos_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    # top_results = torch.topk(cos_scores, k=top_k)

    # st.write("\n\n======================\n\n")
    # st.write("Query:", query)
    # st.write("\nTop 5 most similar sentences in corpus using sentence embedding:")
    #
    # for score, idx in zip(top_results[0], top_results[1]):
    #     st.write("(Score: {:.4f})".format(score))
    #     row_dict = df.loc[df['all_review']== corpus[idx]]
    #     st.write("paper_id:  " , row_dict['hotel_name'] , "\n")
    #     #wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='navy', colormap='rainbow', collocations=False, stopwords = STOPWORDS, mask=mask).generate(corpus[idx])
    #     wordcloud = WordCloud(collocations=False,stopwords=stopwords,background_color='black',max_words=35).generate(corpus[idx])
    #     fig, ax = plt.subplots()
    #     plt.imshow(wordcloud, interpolation='bilinear')
    #     plt.axis("off")
    #     plt.show()
    #     st.pyplot(fig)
    #     st.set_option('deprecation.showPyplotGlobalUse', False)


# embedder = SentenceTransformer('all-MiniLM-L6-v2')
#
# corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
