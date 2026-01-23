#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#************************************
# Auteur : Philippe Mulhem
# Date : January 2023
# Description : Génération d'un vocabulaire à partir d'une liste de fichier texte dans un répertoire
# Usage : python indexer_gendico.py
#************************************

#lib
import os
from nltk.stem.porter import *
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
import string
import codecs
import operator
from operator import itemgetter
import math
import json

#constantes
DOSSIERDOCUMENTS = '/home/emiliostien/Documents/POLYTECH/INFO4/S8/MRI/Projet_RI/TP_Representation/docs/cacm/' # repertoire qui contien la collectionA MODIFIER
FICHMOTSOUTILS = '/home/emiliostien/Documents/POLYTECH/INFO4/S8/MRI/Projet_RI/TP_Representation/common_words.txt' # fichier des mots outils A MODIFIER
FICHVOC = '/home/emiliostien/Documents/POLYTECH/INFO4/S8/MRI/Projet_RI/TP_Representation/Voc.json' # fichier json de sauvegarde du vocabulaire A MODIFIER
FICHINDEX = '/home/emiliostien/Documents/POLYTECH/INFO4/S8/MRI/Projet_RI/TP_Representation/Index.json'
NBDOCS = len(os.listdir(DOSSIERDOCUMENTS)) # nombre de docs total
MOTSOUTILS= {}  # le dictionnaire python des mots outils
voc = {} # le dictionnaire python du vocabulaire, avec les idf
vectors = {}

#Les fonctions

def loaddocFile(filename):
    """
    Lit un fichier et retourne son contenu texte sous forme de chaine (tout en minuscule)
    """
    global DOSSIERDOCUMENTS
    f = open(DOSSIERDOCUMENTS+filename) # ouverture du document
    result = f.read() # lecture de tout le fichier texte d'un seul coup
    f.close()
    return result.lower() # retourne la chaine en minuscule

def chargeMotsOutils(fstopname):
    """
    charge un fichier d'antidictionnaire, qui contient un terme par ligne
    """
    global MOTSOUTILS
    fstop = open(fstopname,'r') # ouverture du fichier
    line = fstop.readline() # lit une ligne (hypothèse : un mot par ligne
    while line:
        MOTSOUTILS[line.strip()]=1 # on crée une entré de dico par terme
        line = fstop.readline()
    fstop.close()

def stringtokenize(chaine):
    """
    Lit une chaine de caractère et renvoie une liste de tokens (mots)
    """
    tokenizer = RegexpTokenizer('[A-Za-z]\\w{1,}') # mot qui commence par une lettre et suite d'au moins un caractère alphanumérique
    return tokenizer.tokenize(chaine)

def filtreMotsOutils(liste):
    """
    Prend en entrée une liste de mots et en filtre les mots outils et retroune la liste nettoyée
    """
    global MOTSOUTILS
    listeResultat = []# liste dans laquelle on va recopier les mots
    for mot in liste:
        if mot not in MOTSOUTILS: #on garde tout ce qui n'est pas un mot outil
            listeResultat.append(mot)
    return listeResultat

def mot2racine(mot):
    """
    Prend en entrée un mot, et renvoie sa racine, calculée à l'aide du PorterStemmer anglais de la librairie nltk
    """
    stemmer = PorterStemmer() # lancement du stemmer
    racine = stemmer.stem(mot) # calcul de la racine du mot
    return racine

def load_vocabulary(filepath):
    with open(filepath,'r') as f:
        return json.load(f)





def get_doc_tf(liste_stem):
    """ 
    Calcule la fréquence de terme (tf) pour un document donné.
    Retourne un dictionnaire {racine: occurrence}
    """
    tf_dict = {}
    for racine in liste_stem:
        tf_dict[racine] = tf_dict.get(racine, 0) + 1
    return tf_dict

def generate_vectors(voc_idf):
    """
    Parcourt la collection et génère le dictionnaire global des vecteurs.
    Structure : { nom_fichier: { terme: tf * idf } }
    """
    vectors = {}
    
    for filename in os.listdir(DOSSIERDOCUMENTS):
        print(f"Indexation vectorielle de : {filename}")
        
        # 1. Prétraitements (Réutilisation des fonctions de indexer_gendico)
        content = loaddocFile(filename)
        tokens = stringtokenize(content)
        clean_tokens = filtreMotsOutils(tokens)
        
        stems = []
        for mot in clean_tokens:
            stems.append(mot2racine(mot))
            
        # 2. Calcul des TF locaux
        tf_local = get_doc_tf(stems)
        
        # 3. Calcul du poids final (tf * idf)
        # On ne stocke que les termes présents (Représentation creuse)
        doc_vector = {}
        for terme, tf_val in tf_local.items():
            if terme in voc_idf:
                # Calcul selon Slide 21: w = tf * idf * nd (avec nd=1)
                weight = tf_val * voc_idf[terme]
                doc_vector[terme] = weight
        
        vectors[filename] = doc_vector
        
    return vectors

def export_index(data, filepath):
    """ Sauvegarde le gros dictionnaire de vecteurs au format JSON """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\nIndex vectoriel exporté avec succès dans : {filepath}")

if __name__ == "__main__":
    # Chargement des ressources nécessaires
    chargeMotsOutils(FICHMOTSOUTILS)
    voc_idf = load_vocabulary(FICHVOC)
    
    # Génération du gros dictionnaire de vecteurs
    print("Démarrage de la génération des vecteurs...")
    all_vectors = generate_vectors(voc_idf)
    
    # Exportation
    export_index(all_vectors, FICHINDEX)
