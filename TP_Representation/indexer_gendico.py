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
NBDOCS = len(os.listdir(DOSSIERDOCUMENTS)) # nombre de docs total
MOTSOUTILS= {}  # le dictionnaire python des mots outils
voc = {} # le dictionnaire python du vocabulaire, avec les idf

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

def listeundoc2voc(liste):
    """
    Lit une liste de racines de termes (d'un document) et modifie le dictionnaire global dico correspondant (mot -> nb de docs).
    Passe par un dictionnaire local qui stocke les termes qui apparaissent, et met ensuite a jour le vocabulaire global.
    """
    global voc
    dicolocal = {}
    for mot in liste: # parcours des racine de la liste
        if mot not in dicolocal.keys(): # si le mot n'est pas déjà une clé du dictionnaire local on affecte à 1
            dicolocal[mot] = 1
    for mot in dicolocal.keys(): # si le mot est déjà une clé du dictionnaire on incrémente sa fréquence
        if mot in voc.keys():
            voc[mot] += 1
        else: # sinon on l'ajoute comme nouvelle clé
            voc[mot] = 1

def rawdocs2voc():
    """
    traitement des fichiers contenus dans DOSSIERDOCUMENTS pour obtenir les termes du vocabulaire dans dico avec leur df
    """
    global DOSSIERDOCUMENTS
    for filename in os.listdir(DOSSIERDOCUMENTS): # parcours des fichiers du répertoire DOSSIERDOCUMENT, à l'aide de la fonction listdir du module os
        print ("Traitement du fichier ",filename)

        #ouverture du fichier
        doccontent = loaddocFile(filename)

        #découpage en mots
        liste = stringtokenize(doccontent)

        #suppression des mots outils
        listeclean = filtreMotsOutils(liste)

        #troncature les mots de la liste
        listeStem = []
        for mot in listeclean:
            listeStem.append(mot2racine(mot))

        #mise a jour du vocabulaire (mot --> fréquence documentaire)
        listeundoc2voc(listeStem)

def idf_voc():
    """
    Modifie le vocabulaire pour passer des df aux idf
    """
    global voc
    for mot in voc.keys(): # parcours des termes du vocabulaire
        voc[mot]=math.log(NBDOCS/float(voc[mot]))

def exportjsonvoc(filenamejson):
    """
    Exporte le dictionnaire sous forme d'un fichier jsoin avec idf
    """
    global voc
    with open(filenamejson, 'w') as fp:
        json.dump(voc, fp)
    fp.close()



if __name__ == "__main__":
    chargeMotsOutils(FICHMOTSOUTILS)
    rawdocs2voc()
    idf_voc()
#    exportDico(d, "mondicoRICM.txt")
    exportjsonvoc(FICHVOC)

