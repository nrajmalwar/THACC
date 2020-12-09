# python training/data_prep.py --input data/train_data/type --output_txt data/train_data/message
import numpy as np
import pandas as pd
import fnmatch, os, argparse, pickle, csv
from bs4 import BeautifulSoup

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--input", required=True,
	help="path to input dataset of emails")
ap.add_argument("-o", "--output_txt", required=True,
    help="path to output txt files")
args = vars(ap.parse_args())

# Consider the new data in excel format with its 'Type' tagged
# Concat all dataframes to single dataframe
# and convert them to list of dataframes with new column Subs
# Remove emails with just urls and extract docs from html files
# Clean messages to get a single line message body

print("Processing data..")

dir = args["input"]
email_data = pd.concat([pd.read_excel(f'{dir}/{f}', usecols = ['Date', 'From Address', 'Subject', 'Message', 'Type']) \
						for f in os.listdir(dir) if f.endswith('.xlsx')], ignore_index=True)
email_data.dropna(inplace=True)
email_data.reset_index(drop=True, inplace=True)
email_data['Message'] = email_data['Message'].str.replace('http\S+|www.\S+', '', case=False)
email_data['Message'] = email_data['Message'].apply(lambda x: BeautifulSoup(x, "lxml").text)
email_data.at[:,'Message'] = email_data['Message'].str.replace('\r\n', ' ')
email_data.at[:,'Message'] = email_data['Message'].str.replace('\n', ' ')
email_data.at[:,'Message'] = email_data['Message'].str.replace('\t', ' ')
email_data.dropna(inplace=True)
email_data.reset_index(drop=True, inplace=True)

# Extract Messages as txt file for NER tagging
for st in ['Finance', 'MaybeUseful']:
	email_data[email_data.Type==st]['Message'].to_csv('temp.csv', index=False, header=False)
	with open(str(args["output_txt"]+'/'+st+'.txt'), "w") as txt_file:
		with open('temp.csv', "r") as csv_file:
			[txt_file.write(" ".join(row)+'\n') for row in csv.reader(csv_file)]
		txt_file.close()
os.remove('temp.csv')
print("Processing Done!")
