# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:15:06 2018

@author: rSanjib
"""
import csv, json, sys




fileInput = '371349589.json'
fileOutput = '371349589.csv'
inputFile = open(fileInput, 'r') #open json file
outputFile = open(fileOutput, 'w') #load csv file
data = json.load(inputFile) #load json content
inputFile.close() #close the input file
output = csv.writer(outputFile) #create a csv.write
output.writerow(data[0].keys())  # header row
for row in data:
    output.writerow(row.values()) #values row