import requests
import json
from datetime import date
import re

import sys

import argparse

import pandas as pd 
import tiktoken
from openai.embeddings_utils import get_embedding
import openai

import pickle

import pandas as pd
import numpy as np

#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, accuracy_score
from sklearn import metrics

from openai.embeddings_utils import plot_multiclass_precision_recall

embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191



## when training the classifier, have it pull from a database that is only filtered to have scored papers
## store the DOI locally (use a flag to determine when you want to udpate the local DOI list.)







## secret token for fh: secret_8ibERlM89As4pGqvLzGqNFjyHQ9yXfEWo1BKdnjSshp
## secret token for local Richard's notion: secret_Tx1aAu8K56ruGDox2o1Zpg7wShZfSRvAtqjkpKbY6M1

## Future house database id: 49788118a4d346b78b589b70fb5af4ab
## Local database id: 0b3a840375e04b9bbe2c2ec98729f132

## currently checks to see if the doi is already in the notion
## but doesn't check for duplicates within the payload.



### get abstracts, titles, authors, affiliation, date, and optionally: tags, journals, using id:
### https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=17284678

today = date.today().strftime("%Y-%m-%d")

openai.api_key = "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" 
token = "secret_8ibERlM89As4pGqvLzGqNFjyHQ9yXfEWo1BKdnjSshp"
notionDatabaseId = "49788118a4d346b78b589b70fb5af4ab" ## notion db
interesting_categories = ["bioengineering","bioinformatics","systems biology","biophysics", "synthetic biology", "cell biology","genetics", "genomics","biochemistry", "molecular biology"]

notionHeaders = {
	"Authorization": "Bearer " + token,
	"Content-Type": "application/json",
	"Notion-Version": "2022-06-28"
}


def query_bioRxiv(published_after,published_before):
	## query something like "https://api.biorxiv.org/details/biorxiv/2022-11-30/2022-11-30"
	print("querying biorxiv for between these dates: " + published_after + " and "+ published_before)
	cursor = 0
	api_endpoint ="https://api.biorxiv.org/details/biorxiv/"+published_after+"/"+published_before+"/"+str(cursor)
	r = requests.get(api_endpoint)
	messages = r.json()['messages']
	collection = r.json()['collection']
	#print(collection)
	print(messages)
	print(api_endpoint)
	while (messages[0]['total'] > 100+cursor):
		cursor += 100
		r = requests.get("https://api.biorxiv.org/details/biorxiv/"+published_after+"/"+published_before+"/"+str(cursor))
		messages = r.json()['messages']
		new_collection = r.json()['collection']
		#print(new_collection)
		collection = collection + new_collection
		print(messages)
	print(api_endpoint)
		#print("not enough so reasking")
		#print(messages)
	#print(collection)
	#print(json.dumps(collection,indent=2))
	#print(json.dumps(collection,indent=2))

	## for each request, check the publication date against the "published since" date and confirm category matches "interesting categories"
	filtered_collection = [x for x in collection if x['category'] in interesting_categories]


	#print(json.dumps(filtered_collection,indent=4))

	for paper in filtered_collection:
		paper_version = int(paper.get("version"))
		if paper_version > 1:

			for i in range(len(filtered_collection)):
				if filtered_collection[i]['doi'] == paper["doi"]:
					del filtered_collection[i]
					break

	for paper in filtered_collection:
		paper["url"] = re.sub('\.source.xml','',paper.get('jatsxml'))
		paper.pop("version")
		paper.pop("license")
		paper.pop("type")
		paper.pop("published")
		paper.pop("server")
		paper.pop("jatsxml")


	with open('./data/biorxiv_db.json', 'w', encoding='utf8') as f: ## saves in notion format, doesn't include abstracts.
		json.dump(filtered_collection, f, ensure_ascii=False)
	#print(json.dumps(filtered_collection,indent=4))
	#print(messages)
	#print(len(filtered_collection))
	#print(type(filtered_collection))
	return filtered_collection
def get_abstract_from_notion(pageId,notionHeaders): ## extremely slow
	print("getting abstract!")
	url = "https://api.notion.com/v1/blocks/"+pageId + "/children"
	res = requests.get(url, headers=notionHeaders)
	print(res.json())
	abstract_text_blocks = res.json()["results"]
	text_blocks= []
	for block in abstract_text_blocks:
		block_text = block["paragraph"]["rich_text"][0]["plain_text"]
		#print_pretty_json(block)
		text_blocks.append(block_text)
		#print(block_text)
		#print("next block:")
	return "\n".join(text_blocks)
def createPage(notionDatabaseId, notionHeaders):
	print("creating page")
	createUrl = 'https://api.notion.com/v1/pages'

	newPageData = {
		"parent": { "database_id": notionDatabaseId},
		"properties": {
			"Title": {
				"title": [
					{
						"text": {
							"content": "Seventh Paper"
						}
					}
				]
			},
			"API Generated": {
				"multi_select": [{"name" : "True"}]
			},
			"Corresponding Author": {
				"rich_text": [
					{
						"text": {
							"content": "Review"
						}
					}
				]
			},
			"URL": {
				"url": "Amazing.com"
			},
			"Authors": {
				"rich_text": [
					{
						"text": {
							"content": "Amazing"
						}
					}
				]
			},
			"Corresponding Author": {
				"rich_text": [
					{
						"text": {
							"content": "Amazing"
						}
					}
				]
			},
			"Category": {
				"rich_text": [
					{
						"text": {
							"content": "Amazing"
						}
					}
				]
			},
			"Abstract": {
				"rich_text": [
					{
						"text": {
							"content": "Amazing"
						}
					}
				]
			},
			"Author Corresponding Institution": {
				"rich_text": [
					{
						"text": {
							"content": "Active"
						}
					}
				]
			}
		}
	}
	
	data = json.dumps(newPageData)
	# print(str(uploadData))

	res = requests.request("POST", createUrl, headers=notionHeaders, data=data)
	print("request sent")
	print(res.status_code)
	print(json.dumps(res.json(),indent=2))
def updatePage(pageId, notionHeaders):
	updateUrl = f"https://api.notion.com/v1/pages/{pageId}"

	updateData = {
		"properties": {
			"Value": {
				"rich_text": [
					{
						"text": {
							"content": "Pretty Good"
						}
					}
				]
			}        
		}
	}
def create_paper_in_notion(notionDatabaseId, notionHeaders,paper):
	createUrl = 'https://api.notion.com/v1/pages'
	newPageData = {
		"parent": { "database_id": notionDatabaseId},
		"properties": {
			"Title": {
				"title": [
					{
						"text": {
							"content": paper["title"]
						}
					}
				]
			},
			"API Generated": {
				"multi_select": [{"name" : "True"}]
			},
			"Corresponding Author": {
				"rich_text": [
					{
						"text": {
							"content": paper["author_corresponding"]
						}
					}
				]
			},
			"Date Uploaded": {
				"rich_text": [
					{
						"text": {
							"content": paper["date"]
						}
					}
				]
			},
			"URL": {
				"url": paper["url"]
			},
			"Authors": {
				"rich_text": [
					{
						"text": {
							"content": paper["authors"]
						}
					}
				]
			},
			"DOI": {
				"rich_text": [
					{
						"text": {
							"content": paper["doi"]
						}
					}
				]
			},
			"Category": {
				"rich_text": [
					{
						"text": {
							"content": paper["category"]
						}
					}
				]
			},
			"Predicted Relevance": {
				"rich_text": [
					{
						"text": {
							"content": str(paper["predicted_relevance"])
						}
					}
				]
			},
			"Abstract": {
				"rich_text": [
					{
						"text": {
							"content": "abstract in document"
						}
					}
				]
			},
			"Author Corresponding Institution": {
				"rich_text": [
					{
						"text": {
							"content": paper["author_corresponding_institution"]
						}
					}
				]
			},
		},
		"children": [
				{
				  "object": "block",
				  "type": "paragraph",
				  "paragraph": {
					"rich_text": [{ "type": "text", "text": { "content": paper["abstract"] } }]
				  }
				}
			  ]
	}

	abstract_paragraphs = break_up_paragraphs(paper["abstract"])
	abstract_paragraphs.insert(0,"Abstract:")
	abstract_blocks = []
	for paragraph in abstract_paragraphs:
		new_block = {
				  "object": "block",
				  "type": "paragraph",
				  "paragraph": {
					"rich_text": [{ "type": "text", "text": { "content": paragraph } }]
				  }
				}
		abstract_blocks.append(new_block)

	newPageData["children"] = abstract_blocks
	data = json.dumps(newPageData)
	#print(str(uploadData))
	print("Adding this paper to Notion: "+ paper["title"])
	res = requests.request("POST", createUrl, headers=notionHeaders, data=data)
	print(res.status_code)
	#print(json.dumps(res.json(),indent=2))
	if res.status_code == 400:
		print("error!! stopping it now.")
		quit()
def update_page(pageId, notionHeaders):
	updateUrl = f"https://api.notion.com/v1/pages/{pageId}"

	updateData = {
		"properties": {
			"Value": {
				"rich_text": [
					{
						"text": {
							"content": "Pretty Good"
						}
					}
				]
			}        
		}
	}
	data = json.dumps(updateData)

	response = requests.request("PATCH", updateUrl, headers=notionHeaders, data=data)

	print(response.status_code)
	print(response.text)
def split_into_sentences(text):
	alphabets= "([A-Za-z])"
	prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
	suffixes = "(Inc|Ltd|Jr|Sr|Co)"
	starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
	acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
	websites = "[.](com|net|org|io|gov)"
	digits = "([0-9])"
	text = " " + text + "  "
	text = text.replace("\n"," ")
	text = re.sub(prefixes,"\\1<prd>",text)
	text = re.sub(websites,"<prd>\\1",text)
	text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
	if "..." in text: text = text.replace("...","<prd><prd><prd>")
	if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
	text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
	text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
	text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
	text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
	text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
	text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
	text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
	if "”" in text: text = text.replace(".”","”.")
	if "\"" in text: text = text.replace(".\"","\".")
	if "!" in text: text = text.replace("!\"","\"!")
	if "?" in text: text = text.replace("?\"","\"?")
	text = text.replace(".",".<stop>")
	text = text.replace("?","?<stop>")
	text = text.replace("!","!<stop>")
	text = text.replace("<prd>",".")
	sentences = text.split("<stop>")
	sentences = sentences[:-1]
	sentences = [s.strip() for s in sentences]
	return sentences
def break_up_paragraphs(original_paragraph):
	paragraphs = original_paragraph.splitlines()
	paragraphs_A = list(filter(None, paragraphs))
	#print("next")
	
	paragraphs_too_long = True
	while paragraphs_too_long:
		paragraphs_B = []
		paragraphs_too_long = False
		for paragraph in paragraphs_A:
			#print(len(paragraph))
			if len(paragraph) > 1800:
				sentences = split_into_sentences(paragraph)
				num_sentences = len(sentences)
				first_paragraph = ' '.join(sentences[0:num_sentences//2])
				last_paragraph = ' '.join(sentences[num_sentences//2:-1])
				paragraphs_B.append(first_paragraph)
				paragraphs_B.append(last_paragraph)
				paragraphs_too_long = True
			else:
				paragraphs_B.append(paragraph)
		if paragraphs_too_long ==True:
			paragraphs_A = paragraphs_B
		else:
			return paragraphs_A
		#print(len(final_paragraphs))
def print_pretty_json(ugly_json):
	print(json.dumps(ugly_json,indent=2))
def translate_notion_to_bioRxiv_format(notionJson, skip_abstracts=False):
	reformatted_db = []
	for i in range(len(notionJson)):

		biorxiv_entry = {}
		entry = notionJson[i]
		#print_pretty_json(entry)
		#biorxiv_entry["abstract_property"] = entry["properties"]["Abstract"]["rich_text"][0]["plain_text"]
		biorxiv_entry["category"] = entry["properties"]["Category"]["rich_text"][0]["plain_text"]
		biorxiv_entry["authors"] = entry["properties"]["Authors"]["rich_text"][0]["plain_text"]
		biorxiv_entry["author_corresponding_institution"] = entry["properties"]["Author Corresponding Institution"]["rich_text"][0]["plain_text"]
		biorxiv_entry["author_corresponding"] = entry["properties"]["Corresponding Author"]["rich_text"][0]["plain_text"]
		biorxiv_entry["doi"] = entry["properties"]["DOI"]["rich_text"][0]["plain_text"]
		biorxiv_entry["url"] = entry["properties"]["URL"]["url"]
		biorxiv_entry["title"] = entry["properties"]["Title"]["title"][0]["plain_text"]
		if entry["properties"]["Date Uploaded"].get("rich_text") != []:
			biorxiv_entry["date"] = entry["properties"]["Date Uploaded"]["rich_text"][0]["plain_text"]
		else:
			biorxiv_entry["date"] = ""
		if entry["properties"]["Relevance Score"].get("select") is not None:
			notion_relevance = entry["properties"]["Relevance Score"].get("select").get("name")
			#print(notion_relevance)
			biorxiv_entry["relevance"] = notion_relevance
		else:
			#print("relevance was none")
			biorxiv_entry["relevance"] = ""

		#biorxiv_entry["abstract"] = get_abstract(entry["id"],notionHeaders)
		if skip_abstracts == True:
			biorxiv_entry["abstract"] = "skipped"
		else:
			biorxiv_entry["abstract"] = get_abstract_from_biorxiv(biorxiv_entry["doi"])
		#print_pretty_json(biorxiv_entry)
		reformatted_db.append(biorxiv_entry)
	return reformatted_db
def get_abstract_from_biorxiv(doi):
	## grabs abstracts from this endpoint: https://api.biorxiv.org/details/biorxiv/10.1101/339747
	url = "https://api.biorxiv.org/details/biorxiv/"+doi
	r = requests.get(url)
	#print(json.dumps(r.json(),indent=2))
	abstract = r.json()["collection"][-1]["abstract"] ## grabs the highest version #, if multiple docs exist for same doi.
	return abstract


def sync_biorxiv_to_notion(published_after=today,published_before=today, refresh_doi_list = True):
	doi_list_filepath = "./data/doi_list.json"
	doi_list = []
	## get list of papers from notion
	if refresh_doi_list:
		converted_notion_db = readDatabase(skip_abstracts = True)

		for paper in converted_notion_db:
			doi_list.append(paper["doi"])
		with open(doi_list_filepath, 'w', encoding='utf8') as f: ## saves in standard biorxiv style json
			json.dump(doi_list, f, ensure_ascii=False)	
	else: 
		with open(doi_list_filepath, 'r', encoding = 'utf8') as f:
			doi_list = json.load(f)

	## get papers from biorxiv	
	papers = query_bioRxiv(published_after,published_before)
	## Remove duplicate DOIs
	paper_dois = []
	for paper in papers:
		paper_dois.append(paper['doi']) ## if there is a duplicate in paper_dois, pick the later version #.	
	

	## check if biorxiv papers are already in notion, remove duplicates.
	non_duplicate_papers = []
	for paper in papers:
		if paper['doi'] not in doi_list:
			non_duplicate_papers.append(paper)
		else:
			print("This DOI is already in notion!" + paper['doi'])
	papers_db_filepath = './data/relevance_db.json'
	with open(papers_db_filepath, 'w', encoding='utf8') as f: ## saves in standard biorxiv style json
		json.dump(non_duplicate_papers, f, ensure_ascii=False)	

	non_duplicate_papers, papers_with_predictions_db_filepath = predict_relevance(papers_without_predictions_db_filepath = papers_db_filepath) ## add predicted relevance.

	add_papers_to_notion(papers_with_predictions_db_filepath)


def add_papers_to_notion(paper_db_filepath='./data/papers_with_predictions_db.json'):
	with open(paper_db_filepath, 'r', encoding ='utf8') as f:
		papers = json.load(f)
	for paper in papers:
		create_paper_in_notion(notionDatabaseId, notionHeaders,paper)

def readDatabase(skip_abstracts=False, only_scored_papers = False):
	readUrl = f"https://api.notion.com/v1/databases/{notionDatabaseId}/query"
	print("requesting notion db")
	if only_scored_papers:
		data = { 
			"filter": {
				"property": "Relevance Score",
				"select": {
					"is_not_empty":True
					}
				}
			}
	else:
		data = {}


	res = requests.request("POST", readUrl, headers=notionHeaders, data = json.dumps(data))
	response = res.json()
	#print(response)
	results = response["results"]
	while response['has_more']:
		print("getting more data...")
		data['start_cursor'] = response['next_cursor']
		res = requests.request("POST", readUrl, headers=notionHeaders,data = json.dumps(data))
		response = res.json()
		#print_pretty_json(response)
		results = results + response["results"]

	#print(json.dumps(res.json(),indent=2))

	with open('./data/notion_db.json', 'w', encoding='utf8') as f: ## saves in notion format, doesn't include abstracts.
		json.dump(results, f, ensure_ascii=False)

	revised_db = translate_notion_to_bioRxiv_format(results,skip_abstracts)
	if skip_abstracts:
		print("not collecting abstracts, not updating the local db json.")
	else:
		with open('./data/biorxiv_from_notion_db.json', 'w', encoding='utf8') as f: ## saves in standard biorxiv style json
			json.dump(revised_db, f, ensure_ascii=False)	
	print("notion database fully downloaded.")
	return revised_db

def create_training_openai_dataset(json_dataset = "data/biorxiv_from_notion_db.json"):
	print("preparing dataset.")
	# embedding model parameters
	# load & inspect dataset
	df = pd.read_json(json_dataset)
	df = df[["doi", "title", "authors", "author_corresponding", "author_corresponding_institution","relevance", "date","category","abstract","url"]]
	df = df.dropna()
	df["combined"] = (
	    "Title: " + df.title.str.strip() + 
	    "; Authors: "+ df.authors.str.strip() + 
	    "; Corresponding Author: "+ df.author_corresponding.str.strip() + 
	    "; Author Institution: "+ df.author_corresponding_institution.str.strip() + 
	    "; Category: "+ df.category.str.strip() + 
	    "; Abstract: " + df.abstract.str.strip()
	)
	#df.head(2)
	top_n = 1000
	df['relevance'].replace('', np.nan, inplace=True)
	df = df.dropna()

	df['relevance'].replace('5 Extremely Relevant', 5, inplace=True) ## this is here because there aren't enough!
	df['relevance'].replace('4 Very Relevant', 4, inplace=True) ## this is here because there aren't enough!
	df['relevance'].replace('3 Possibly Relevant', 3, inplace=True) ## this is here because there aren't enough!
	df['relevance'].replace('2 Probably Irrelevant', 2, inplace=True) ## this is here because there aren't enough!
	df['relevance'].replace('1 Certainly Irrelevant', 1, inplace=True) ## this is here because there aren't enough!
	

	df = df.tail(top_n*2)
	print("encoding embedding for combined text...")
	encoding = tiktoken.get_encoding(embedding_encoding)
	df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
	df = df[df.n_tokens <= max_tokens].tail(top_n)
	len(df)
	df.to_csv("data/regressor_openai_embedded_dataset_no_embeddings.csv") # this isn't used, just an intermediate saved state.
	#print(df)
	return df

def create_regression_openai_dataset(json_dataset = "data/biorxiv_db.json"):
	print("preparing dataset.")
	# embedding model parameters
	# load & inspect dataset
	df = pd.read_json(json_dataset)
	df = df[["doi", "title", "authors", "author_corresponding", "author_corresponding_institution", "date","category","abstract","url"]]
	df = df.dropna()
	df["combined"] = (
	    "Title: " + df.title.str.strip() + 
	    "; Authors: "+ df.authors.str.strip() + 
	    "; Corresponding Author: "+ df.author_corresponding.str.strip() + 
	    "; Author Institution: "+ df.author_corresponding_institution.str.strip() + 
	    "; Category: "+ df.category.str.strip() + 
	    "; Abstract: " + df.abstract.str.strip()
	)

	top_n = 1000
	df = df.dropna()

	df = df.tail(top_n*2)
	print("encoding embedding for combined text...")
	encoding = tiktoken.get_encoding(embedding_encoding)
	df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
	df = df[df.n_tokens <= max_tokens].tail(top_n)
	len(df)
	df.to_csv("data/regressor_openai_embedded_dataset_no_embeddings.csv") # this isn't used, just an intermediate saved state.
	#print(df)
	return df
def add_openai_embeddings_to_dataframe(df):
	print("adding embeddings!")
	df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))
	df.to_csv("data/regressor_openai_embedded_dataset.csv")
	return df
def train_random_forest_classifier(random_state=40):
	print("training classifier")
	datafile_path = "data/regressor_openai_embedded_dataset.csv"
	df = pd.read_csv(datafile_path)

	df["embedding"] = df.embedding.apply(eval).apply(np.array)  # convert string to array
	#print(df)
	# split data into train and test
	X_train, X_test, y_train, y_test = train_test_split(
	    list(df.embedding.values), df.relevance, test_size=0.2, random_state=random_state
	)

	# train random forest classifier
	clf = RandomForestRegressor(n_estimators=100, random_state = random_state)
	clf.fit(X_train, y_train)

	# save classifier
	file = open('classifier_regressor.pkl','wb')
	pickle.dump(clf, file)
	file.close()
	preds = clf.predict(X_test)
	#probas = clf.predict_proba(X_test)

	#report = classification_report(y_test, preds)
	#print(report)
	#print(y_train)
	#print(y_test)
	print(preds)
	print(y_test.tolist())
	#print(probas)

	pred = preds 
	gt = y_test.tolist()

	print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(gt, pred))
	print('Mean Squared Error (MSE):', metrics.mean_squared_error(gt, pred))
	print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(gt, pred)))
	mape = np.mean(np.abs((gt - pred) / np.abs(gt)))
	print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
	print('Accuracy:', round(100*(1 - mape), 2))

	#plot_multiclass_precision_recall(probas, y_test, [1, 2, 3, 4, 5], clf)

	return clf, X_train, X_test, y_train, y_test

def predict_relevance(papers_without_predictions_db_filepath = './data/relevance_db.json',papers_with_predictions_db_filepath = './data/papers_with_predictions_db.json'): ## call this while adding new papers into notion page.
	with open(papers_without_predictions_db_filepath, 'r', encoding='utf8') as f:
		papers = json.load(f)


	df = create_regression_openai_dataset(papers_without_predictions_db_filepath)
	df = add_openai_embeddings_to_dataframe(df)
	#print(df)
	#df["embedding"] = df.embedding.apply(eval).apply(np.array)  # convert string to array
	embeddings = list(df.embedding.values)

	f = open('classifier_regressor.pkl','rb')
	clf = pickle.load(f)
	f.close()

	predictions = clf.predict(embeddings)

	## match those predictions back up to the original DOIs / entries

	## add those predictions to  the notion db. 
	#print(predictions)

	for i in range(len(papers)):
		papers[i]['predicted_relevance'] = predictions[i]
	with open(papers_with_predictions_db_filepath, 'w', encoding='utf8') as f: ## saves in standard biorxiv style json
		json.dump(papers, f, ensure_ascii=False)	
	return papers, papers_with_predictions_db_filepath


def train_regressor():
	readDatabase(only_scored_papers=True)
	df = create_training_openai_dataset()
	add_openai_embeddings_to_dataframe(df)
	train_random_forest_classifier()
	

def main():
	parser = argparse.ArgumentParser(description = "refresh_doi_list=False if you don't want to avoid duplicate papers.")
	parser.add_argument('--train', '-t', action = 'store_true', help = 'train flag. will download the current notion db and replace the classifier stored in this repo with a new regression classifier.')
	parser.add_argument('--upload', '-u', action = 'store_true', help = 'upload flag. will download the current notion db to check for duplicates, then download biorxiv through the specified date range. then it will predict relevance using openai embedding / classifier, and upload the nonduplicate papers to notion with those relevance predictions included.')
	parser.add_argument('dates', nargs = 2, type=str)
	parser.add_argument('--no_doi_refresh', '-d', action = 'store_true', help = 'skips the notion download step to check for duplicates. faster but you will get duplicates using this flag unless your date range is not already included in the db')
	
	args = parser.parse_args()
	if args.train:
		train_regressor()
	elif args.upload:
		start_date = args.dates[0]
		end_date = args.dates[1]
		if args.no_doi_refresh:
			refresh_flag = False
		else:
			refresh_flag = True
		sync_biorxiv_to_notion(start_date, end_date, refresh_doi_list = refresh_flag)
	else:
		print("you input something wrong. try the -h flag")

### The real 2 good functions
if __name__ == "__main__":
	main()
#train_regressor()
#sync_biorxiv_to_notion("2023-01-05","2023-01-05", refresh_doi_list = True)

## some helper step functions.
#query_bioRxiv("2023-01-02","2023-01-02")
#readDatabase(skip_abstracts=True, only_scored_papers = True)
#train_random_forest_classifier(99)
#print(predict_relevance())
#predict_relevance()
#add_papers_to_notion()

#with open('./notion_db.json','r',encoding='utf8') as f:
#	notion_db = json.load(f)
#translate_notion_to_bioRxiv_format(notion_db,True)
## fine tune model using title, authors, author_corresponding, author_corresponding_institution, category, abstract that autocompletes "interest score: X" based on some representative scores.

## or do an embedding of the abstracts, then train sklearn random forest classifier (see gpt3)

## for each request that passes the above criteria, prompt gpt3 to see if it would be interesting. 
## include server, institution name, abstract, category





## grab the corresponding author email and name out of the xml?