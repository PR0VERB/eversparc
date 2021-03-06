# #### ToDo: Entity Extraction for graph...
# Correlations
# 
# I will continue to do work like this regardless of the industry where I end up in 2,5,10 years. Just because I want to be a thought leader, it doesn't mean I should abandon my calling to create or facilitate the creation of tools. I believe that life is like a lottery, you have to play if you want to win. That doesn't mean that you have to be as seemingly directionless as I am. I am only doing things this way because I have set for myself goals that seem unattainable in this industry.
# Only return relevant to country...

# In[30]:


import pandas as pd
# import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import numpy as np
import spacy
# import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
nltk.download('stopwords')
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics.pairwise import linear_kernel
# import camelot as cmt
import re
import string
import streamlit as st
# import tabula as tbl
# import html5lib
import copy
import en_core_web_sm
# Ideas: Check knowledge graph then use that to create search engine with the cosine similarity -- 
# Can extract entities and check which are most related using cos
# TODO: rank south africa first,
#       give suggestions to click instead of returning first result
# countryData = pd.read_csv('countryData.csv')
# st.dataframe(countryData)

# df_ = pd.read_csv('86077x278_1Jan2020.csv')
# st.dataframe(df_.loc[:,:10])


def main():

    menu = ['Home', 'About']
    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Home':
        st.subheader('Home')
        homePage()
    else:
        st.write('Welcome to the about page')

def homePage():

    b1 = st.button('yay')
    b2 = st.button('yo')

    if b1: # https://discuss.streamlit.io/t/how-to-add-a-function-to-a-button/1478/2
        st.dataframe(df_new)

        st.line_chart(df_new)

    if b2: 
        st.dataframe(df_new)

        st.line_chart(df_new)


@st.cache(allow_output_mutation=True)
def loadNLP():
#     nlp_ = spacy.load('en_core_web_sm')
    nlp_ = en_core_web_sm.load()
    return nlp_

nlp = loadNLP()

@st.cache # from streamlit cheat sheet https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py
def dfUnmodified(): #put all the dataframe ops in cached function
    df_ = pd.read_csv('86077x278_1Jan2020.csv')


    df = df_.copy()
#     df['Dates'] = pd.to_datetime(df['Dates'])
    df = df.sort_values('Dates')

    # Country Names
    countryData = pd.read_csv('countryData.csv')
    countryNames = countryData['Country (or dependency)'].values
    countryNames = np.append(countryNames, ['Dates', 'OECD', 'G20', 'G7','Euro', 'Korea', 'Republic', 'USA', 'UK'])

    # Finally, the renaming
    dfCols = df.columns.values

    saOld = []
    saNew = []
    for col in dfCols:
        if not any(word in col for word in countryNames):
            saOld.append(col)
            col = col+' South Africa'
            saNew.append(col)

    df = df.rename(columns = dict(zip(saOld, saNew)))

    return df


df_unmodified_ = dfUnmodified()
df_unmodified = copy.deepcopy(df_unmodified_)
dfCols = df_unmodified.columns.values

# df_ = pd.read_csv(r'C:\Users\bseot\Documents\REAL_ESTATE\QUANTEC\ALY\QUANTEC_30DEC2020\QUANTEC_86077x278_1Jan2020.csv')


# df = df_.copy()


@st.cache
def nouns(sent):
    sent = sent.lower()
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    sent = [tok[0] for tok in sent if tok[1] == 'NN']    
    return sent

# Crude Matching

@st.cache
def MatchingWords(phrase, otherItems):
    
    '''
    Find the items from otherItems most similar to phrases, based on frequency of the same word. 
    
    phrases: array of string types. From user input
    otherItems: array of string types. From corpus.
    '''
    phrase = nouns(phrase)
    phrase = set(phrase)
    SIMS_ITEMS = [] #for the rank of phrases to match in the tuple we will print
    for i, item in enumerate(otherItems):
        item = item.lower()
        item = set(item.split())
        phraseItem = phrase.intersection(item)
        sim = float(len(phraseItem))
        realPhrase = df_unmodified.iloc[:, [i]].columns.values[0]
        SIMS_ITEMS.append([realPhrase, sim, i])
        
    SIMS_ITEMS = sorted(SIMS_ITEMS, key = lambda i: -i[1])   
    # return all of the items, even if it has no country name. see countryVerifier()
    return SIMS_ITEMS


# The code below finds the item indices

@st.cache
def countryVerifier(phrase, otherItems):
    SIMS_ITEMS = MatchingWords(phrase, otherItems)
    finalPhrases = [phras[0] for phras in SIMS_ITEMS]

    print(finalPhrases)
#     iteratedFirst = iter(set(countryNames).intersection(set(finalPhrases[0].split()))) # doesn't work with South Africa
#     countr = next(iteratedFirst, None)
    doc = nlp(SIMS_ITEMS[0][0])
    countr = [(X.text) for X in doc.ents][0]
    countr = countr.translate(str.maketrans('', '', string.punctuation)).strip()
    countr = set(countryNames).intersection({countr})
    countr = list(countr)[0]
#     countr = next(iteratedFirst, None)
    print(countr)
    tmp = [] #array that will store the phrases that have the countries
    for phras in finalPhrases:
        print((phras))
        if countr in phras:
            print(countr)

            tmp.append(phras)
    NEW_SIMS_ITEMS = []
    for item in SIMS_ITEMS:
        if item[0] in tmp:
            NEW_SIMS_ITEMS.append(item)
    return NEW_SIMS_ITEMS, finalPhrases



# ## Correlations for the next 8 results

# In[19]:

@st.cache
def pearsonCorr(phrase, otherItems):
    semanticSuggestions = countryVerifier(phrase, otherItems)
    semanticSuggestions = [item[0] for item in semanticSuggestions[0]]
    firstSuggestion = semanticSuggestions[0]
    correlations = df_unmodified[semanticSuggestions].corrwith(df[firstSuggestion])
    return correlations
#     return f'{firstSuggestion} \n {twoToNine}'


def dfNew(columns):
    '''
    columns: array of columns
    '''
    # Mutate bar
    df_new = df_unmodified
    df_new_index = df_new.Dates
    df_new = df_new.drop(['Dates'], axis = 1)
    df_new.index = df_new_index
    df_new = df_new[columns]
    df_new = df_new.dropna(how = 'all', axis = 0)
    return df_new

query = st.text_input('Enter some text')
matched = MatchingWords(query, dfCols)
df_new = dfNew([matched[0][0]])

# result = pearsonCorr(query, dfCols)
# result = result.sort_values(ascending = False).drop_duplicates()
# st.dataframe(df_unmodified.iloc[:20,:30]) # works

if __name__ == '__main__':
    main()
