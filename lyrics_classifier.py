#!/usr/bin/env python
# coding: utf-8
# # Project: Lyrics Classifier

# Workflow
# 1- Get lyrics from the web
# 2- Clean the list of songs (with spacy)
# 3- Vectorize the clean list
# 4- Transform the list
# 5- Apply the model (NB, Logistic Regression, etc..)



import requests
from bs4 import BeautifulSoup as soup
import numpy as np
import os
import pandas as pd
import spacy
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


# --------------------------------------------------------
# Class
# -------------------------------------------------------
class Artist:
    def __init__(self, name):
        self.name = name
        self.newname = ""
        self.web = ""
        self.song = ""
        self.songtext = ""
        self.title = ""


    def get_links_artist(self):
        """
            Find all the links for a given artist where songs can be found and store the links in a list
        """
        page_list=[]

        # Adjust the name of the artist
        artist_name = self.name.lower()

        if artist_name.startswith("the"):
            artist_name = artist_name.replace('the','')
            artist_name = artist_name.replace(' ','-')
            artist_name = artist_name[1:]
        else:
            artist_name = artist_name.replace(' ','-')

        self.newname = artist_name

        # Look for the lyrics in metrolyrics.com
        link_str = 'https://www.metrolyrics.com/' + artist_name + '-lyrics.html'
        artist_request = requests.get(link_str)
        artist_soup = soup(artist_request.text,'html.parser')
        page_tag = artist_soup.find_all(attrs={"class":"pages"})

        if len(page_tag) == 0:
            page_tag = artist_soup.find_all('a',attrs={"class":"active"})
            page_list.append(page_tag[0].get('href'))
        else:
            for each in page_tag[0].find_all('a'):
                #print(each.get('href'))
                page_list.append(each.get('href'))

        file_name = "Songs/" + artist_name + "_link_pages.csv"
        np.savetxt(file_name, page_list,delimiter=",", fmt='%s')

        self.web = self.newname + "_link_pages.csv"

        return None


    def get_links_songs(self):
        page_list = []

        for line in open("Songs/"+ self.web):
            csv_row = line.split()

            art = requests.get(csv_row[0])
            art_soup = soup(art.text,'html.parser')
            song_tag = art_soup.find_all('tbody')

            for each in song_tag[0].find_all('a',attrs={"class":"title hasvidtable"}):
                page_list.append(each.get('href'))

            for each in song_tag[0].find_all('a',attrs={"class":"title"}):
                page_list.append(each.get('href'))


        file_name = "Songs/" + self.newname + "_link_songs.csv"
        np.savetxt(file_name, page_list,delimiter=",", fmt='%s')

        self.song = self.newname + "_link_songs.csv"

        return None


    def get_song_lyrics(self):
        """
            Read the file with the song links, read the text of the song and save it in a list
            Save also a list with the titles of the songs
        """
        song_list = []
        title_list = []

        for line in open("Songs/" + self.song ):
            csv_row = line.split()

            art = requests.get(csv_row[0])
            art_soup = soup(art.text,'html.parser')
            song_tag = art_soup.find_all(attrs={"class":"lyrics-body"})
            mySong = ""
            for each in song_tag[0].find_all('p',class_="verse"):

                mySong = mySong + " " + each.text.replace('\n', ' ')

            #Find the title
            myTitle = art_soup.find_all('h1')

            mySong.strip()
            if (mySong.find('instrumental') == -1):
                song_list.append(mySong)
                title_list.append(myTitle[0].text)

        #Save the song text in a file
        df = pd.DataFrame(data=song_list)
        file_name = "Songs/" + self.newname + "_songs_text.csv"
        df.to_csv(file_name, sep=',',index=False,header=None)

        #Save the song titles in a file
        df = pd.DataFrame(data=title_list)
        file_name = "Songs/" + self.newname + "_songs_title.csv"
        df.to_csv(file_name, sep=',',index=False,header=None)

        self.songtext = self.newname + "_songs_text.csv"
        self.title = self.newname + "_songs_title.csv"

        return None


# ---------------------------------------------------------------
# ## General functions
# --------------------------------------------------------------
def create_Artist(artist_name):
    a = Artist(artist_name)
    a.get_links_artist()
    a.get_links_songs()
    a.get_song_lyrics()
    return a


def clean_song_list(reg_list,model):
    """
        Input a list of strings - in this case the songs - and return the clean list of songs
    """
    clean_song_list = []

    for string in reg_list:
        doc = model(string)
        clean_text = ''
        for word in doc:
            if not word.is_stop and word.lemma_ != '-PRON-' and word.pos_ != 'PUNCT':
                word = word.lemma_
                clean_text += word + ' '

        clean_song_list.append(clean_text)

    return clean_song_list


def create_df_artist_song(art):
    """
        Read the text of all songs saved in csv files and create a DataFrame with text song and titles
    """
    file_name_text = "Songs/" + art.songtext
    df_text = pd.read_csv(file_name_text,names=["Text"])

    file_name_title = "Songs/" + art.title
    df_title = pd.read_csv(file_name_title,names=["Title"])
    #print(df_text.shape)
    #print(df_title.shape)

    df = pd.concat([df_title, df_text],axis=1)
    df.dropna(inplace=True)
    df['Artist'] = art.name
    #print(df.shape)
    return df


def create(name,art_nr):
    artist = create_Artist(name)
    #print(artist.name)

    df = create_df_artist_song(artist)
    #print("create df artist song")
    file_name = "Songs/all_artists_songs.csv"
    if art_nr < 1:
        df.to_csv(file_name, sep=',',index=False)
    else:
        df_all = pd.read_csv(file_name,names=['Title','Text','Artist'])
        os.remove(file_name)
        df2 = pd.concat([df_all, df])
        #Save again to csv
        df2.to_csv(file_name, sep=',',index=False,header=None)
    return




# --------------------------
# __main__
# --------------------------


if __name__ =="__main__":

    all_artists = []
    all_artists_df = pd.DataFrame()

    if os.path.exists("Songs/all_artists_songs.csv"):
        os.remove("Songs/all_artists_songs.csv")

    while True:
        data = input("Enter an artist: \n")
        if len(data)==0:
          #print("No more artists")
          break
        else:
            all_artists.append(data)

    check_text = input("Enter a text and I'll predict the artist: \n")

    print(".....\n")
    print("Program starts...")

    # convert input into a list
    list_check_text = [check_text]

    # Create all my artist
    print("Creating artists...")

    final_artist_list=[]
    for l in all_artists:
        try:
            create(l,all_artists.index(l))
            final_artist_list.append(l)
        except Exception as e:
            print(f"Can't find songs from {l}")

    # Concatenate all artists df--> done in my function
    #df_all = pd.concat(final_artist_list)
    # Create a df with all songs of all artists
    df_all = pd.read_csv("Songs/all_artists_songs.csv",names=['Title','Text','Artist'],skiprows=1)
    df_all.set_index('Title',inplace=True)


    #Convert text lyrics column into list
    list_text = df_all['Text'].tolist()
    print("Cleaning song lyrics..")
    # Clean song lyrics
    model = spacy.load('en_core_web_md')
    new_clean=clean_song_list(list_text,model)

    # Create new df with title, clean text and artist number
    df_clean = pd.DataFrame(index=df_all.index)
    df_clean['Artist'] =df_all['Artist']
    df_clean['Text'] = new_clean
    #art_fac = pd.factorize(df_all['Artist'])
    #df_clean['Artist_Cat']=art_fac[0]

    # ## The Tf-Idf Transformer

    pipeline = make_pipeline(
                CountVectorizer(),
                TfidfTransformer(),
                MultinomialNB() #alpha=0.0000001 --> unigeness of words super important
                )

    #y = [a1.name] * df1.shape[0] + [a2.name]  * df2.shape[0]
    y = df_clean['Artist']
    X = np.array(df_clean['Text'])

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    print("Fitting the model...")
    pipeline.fit(X_train,y_train)
    #print(f"My model score is: {pipeline.score(X_test,y_test).round(2)}")

    #pipeline.fit(X,y)
    print("Predicting artist...\n")
    result = pipeline.predict_proba(list_check_text)
    #print(result)
    print(f"The artist is: {final_artist_list[result.argmax()]}")
