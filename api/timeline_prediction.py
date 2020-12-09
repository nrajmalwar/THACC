import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os, fnmatch, argparse, pickle, random, re
from bs4 import BeautifulSoup
import spacy

#-----------------------------------Level 1 -----------------------------------#

def types_classifier(dir :str, classifier : str, tokenizer :str):
	# Provide the directory/folder which contains all your csv files
	print("Loading all excel files in the dataset..")
	# Get all the csv path in a list
	# Concat all dataframes to single dataframe
	# and convert them to list of dataframes with new column Subs
	# Remove emails with just urls and extract docs from html files
	# Clean messages to get a single line message body
	email_data = pd.concat([pd.read_excel(f'{dir}/{f}', usecols = ['Date', 'From Address', 'Subject', 'Message']) \
							for f in os.listdir(dir) if f.endswith('.xlsx')], ignore_index=True)
	email_data.dropna(inplace=True)
	email_data.reset_index(drop=True, inplace=True)
	email_data['Date']= pd.to_datetime(email_data.Date, dayfirst = True)
	email_data['Subs'] = email_data['From Address'].map(str) + ' ' + email_data['Subject'].map(str)
	email_data['Message'] = email_data['Message'].str.replace('http\S+|www.\S+', '', case=False)
	email_data['Message'] = email_data['Message'].apply(lambda x: BeautifulSoup(x, "lxml").text)
	email_data.at[:,'Message'] = email_data['Message'].str.replace('\r\n', ' ')
	email_data.at[:,'Message'] = email_data['Message'].str.replace('\n', ' ')
	email_data.at[:,'Message'] = email_data['Message'].str.replace('\t', ' ')
	email_data.dropna(inplace=True)
	email_data.reset_index(drop=True, inplace=True)
	MAXLEN = 50
	print("loading types classsifer model and tokenizer...")
	# load trained type classifier model
	model = load_model(classifier)
	# load Tokenizer
	tokenizer = pickle.load(open(tokenizer, 'rb'))

	# integer encode documents
	encoded_text_pred = tokenizer.texts_to_sequences(email_data.Subs.values)
	encoded_text_pred = pad_sequences(encoded_text_pred, maxlen=MAXLEN, dtype='int32', value=0)

	# Predict on emails
	print("Predicting on emails..")
	labels = model.predict(encoded_text_pred, batch_size=1000, workers=-1, use_multiprocessing=True)
	email_data['Type'] = [np.argmax(i) for i in labels]
	LABEL_MAP = { 0 : "Finance", 1 : "MaybeUseful", 2 : "NotFinance"}
	email_data.loc[:, 'Type'] = email_data.Type.map(LABEL_MAP)

	print("Classifying emails..")
	print("\n",email_data.Type.value_counts(), "\n")

	#for st in ['Finance', 'MaybeUseful']
	finance_data = email_data[email_data.Type=='Finance'][['Date','Message']]
	#maybeuseful_data = email_data[email_data.Type=='MaybeUseful'][['Date', 'Message']]

	return finance_data

#-----------------------------------Level 2 -----------------------------------#

def entities_extractor(ner_model: str):

	finance_data = types_classifier(dir = 'data/test_data/',
									classifier = 'weights/level1/model.h5',
									tokenizer = 'weights/level1/tokenizer.pkl')

	finance_data.reset_index(drop=True, inplace=True)
	# Prediction on Messages to create columnns
	info={}
	print("Loading NER model..")
	nlp = spacy.load(ner_model)
	print("Predicting Labels and Info for Financial emails...")
	# Adding columns for every labels with respective values if present
	for i, doc in enumerate(nlp.pipe(finance_data.Message, batch_size=500)):
		for ent in doc.ents:
			finance_data.at[i, ent.label_] = ent.text
	finance_data.fillna(0, inplace=True)

	return finance_data

#-----------------------------------Level 3 Work Needed------------------------#

def generate_timeline():
	timeline_data = entities_extractor(ner_model = 'weights/level2/')
	timeline_data.reset_index(drop=True, inplace=True)
	# extract banking and non-banking data
	print("Segregating Bank and Non-bank data..")
	df_bank = timeline_data[(timeline_data['Bank Name']! = 0) &
					(timeline_data['Transaction Amount']! = 0) &
					(timeline_data['Transaction Out']! = 0) &
					(timeline_data['Transaction In']! = 0)]
	df_nonbank = timeline_data.drop(df_bank.index)
	df_bank.sort_values(by='Date', inplace=True)
	df_bank.reset_index(drop=True, inplace=True)
	df_nonbank.sort_values(by='Date', inplace=True)
	df_nonbank.reset_index(drop=True, inplace=True)

	# Regex Amount
	print("Generating Timeline..")
	def str2float(txt):
		x = re.findall("(\d+(?:\.\d+)?)", txt.replace(',', ''))
		f = float(np.array(x)) if len(x) == 1 else 0.00
		return f

	df_bank['Transaction Amount'] = df_bank.Amount.apply(str2float)
	df_nonbank['Transaction Amount'] = df_nonbank.Amount.apply(str2float)

	# Group Banking data with non-banking data based on amount and info
	for i in range(len(df_nonbank)):
		for j in range(len(df_bank)):
			if ((df_nonbank.at[i, 'Amount'] == df_bank.at[j, 'Amount']) and (df_nonbank.Date[i].date == df_bank.Date[j].date))):
				#df_bank.at[j, 'Info'] = list(df_nonbank.at[i,'Info'].union(df_bank.at[j, 'Info']))
				#df_bank.at[j, 'Subtype'] = str(df_bank.at[j, 'Subtype'] + ' ' + df_nonbank.at[i, 'Subtype'])

	# Create timeline
	def total(i):
		if (i == 0):
			return (np.round(- df_bank.at[0, 'Transaction Amount'], 2) if ('Debit' in df_bank.at[i, 'Transaction Out'])
														else np.round(df_bank.at[0, 'Amount'], 2))
		else:
			return (np.round(total(i-1) - df_bank.at[i, 'Transaction Amount'], 2) if ('Debit' in df_bank.at[i, 'Transaction Out'])
														else np.round(total(i-1) + df_bank.at[i, 'Transaction Amount'], 2))

	df_bank['Total'] = [total(i) for i in range(df_bank.shape[0])]

	# Generate Output CSV files
	df_bank.to_csv('data/history.csv', index=False)
	print("Timeline Generated!")
	print(df_bank.head())
#-----------------------------------End ---------------------------------------#
