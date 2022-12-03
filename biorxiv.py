import requests
import json
from datetime import date
import re

## getting abstracts out of notion takes way too long. 
## grab abstracts from this endpoint instead: https://api.biorxiv.org/details/biorxiv/10.1101/339747

today = date.today().strftime("%Y-%m-%d")

token = "secret_Tx1aAu8K56ruGDox2o1Zpg7wShZfSRvAtqjkpKbY6M1"
notionHeaders = {
	"Authorization": "Bearer " + token,
	"Content-Type": "application/json",
	"Notion-Version": "2022-06-28"
}


notionDatabaseId = "0b3a840375e04b9bbe2c2ec98729f132"


interesting_categories = ["biophysics", "synthetic biology", "cell biology","genetics", "genomics","biochemistry", "molecular biology"]
sometimes_interesting_categories = ["bioengineering","bioinformatics","systems biology"]




def query_bioRxiv(published_after,published_before):
	## query something like "https://api.biorxiv.org/details/biorxiv/2022-11-30/2022-11-30"
	print("querying biorxiv for between these dates: " + published_after + " and "+ published_before)
	cursor = 0
	r = requests.get("https://api.biorxiv.org/details/biorxiv/"+published_after+"/"+published_before+"/"+str(cursor))
	messages = r.json()['messages']
	collection = r.json()['collection']
	#print(collection)
	#print(messages)

	while (messages[0]['total'] > 100+cursor):
		cursor += 100
		r = requests.get("https://api.biorxiv.org/details/biorxiv/"+published_after+"/"+published_before+"/"+str(cursor))
		messages = r.json()['messages']
		new_collection = r.json()['collection']
		#print(new_collection)
		collection = collection + new_collection
		#print("not enough so reasking")
		#print(messages)
	#print(collection)
	print(json.dumps(collection,indent=2))
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



	#print(json.dumps(filtered_collection,indent=4))
	#print(messages)
	#print(len(filtered_collection))
	#print(type(filtered_collection))
	return filtered_collection



def readDatabase(databaseId, notionHeaders):
	readUrl = f"https://api.notion.com/v1/databases/{databaseId}/query"
	print("requesting notion db")
	res = requests.request("POST", readUrl, headers=notionHeaders)
	data = res.json()

	print(res.status_code)
	#print(json.dumps(res.json(),indent=2))

	with open('./notion_db.json', 'w', encoding='utf8') as f: ## saves in notion format, doesn't include abstracts.
		json.dump(data, f, ensure_ascii=False)

	revised_db = translate_notion_to_bioRxiv_format(data["results"])
	with open('./biorxiv_from_notion_db.json', 'w', encoding='utf8') as f:
		json.dump(data, f, ensure_ascii=False)	
	print("notion database fully downloaded.")
def get_abstract_from_notion(pageId,notionHeaders):
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

def createPage(databaseId, notionHeaders):
	print("creating page")
	createUrl = 'https://api.notion.com/v1/pages'

	newPageData = {
		"parent": { "database_id": databaseId},
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
def create_paper_in_notion(databaseId, notionHeaders,paper):
	createUrl = 'https://api.notion.com/v1/pages'

	newPageData = {
		"parent": { "database_id": databaseId},
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
	# print(str(uploadData))
	print("Adding this paper to Notion: "+ paper["title"])
	res = requests.request("POST", createUrl, headers=notionHeaders, data=data)
	print(res.status_code)
	print(json.dumps(res.json(),indent=2))
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
def translate_notion_to_bioRxiv_format(notionJson):
	reformatted_db = notionJson
	for i in range(len(notionJson)):

		biorxiv_entry = {}
		entry = notionJson[i]
		#print_pretty_json(entry)
		biorxiv_entry["abstract_property"] = entry["properties"]["Abstract"]["rich_text"][0]["plain_text"]
		biorxiv_entry["category"] = entry["properties"]["Category"]["rich_text"][0]["plain_text"]
		biorxiv_entry["authors"] = entry["properties"]["Authors"]["rich_text"][0]["plain_text"]
		biorxiv_entry["author_corresponding_institution"] = entry["properties"]["Author Corresponding Institution"]["rich_text"][0]["plain_text"]
		biorxiv_entry["author_corresponding"] = entry["properties"]["Corresponding Author"]["rich_text"][0]["plain_text"]
		biorxiv_entry["doi"] = entry["properties"]["DOI"]["rich_text"][0]["plain_text"]
		biorxiv_entry["url"] = entry["properties"]["URL"]["url"]
		biorxiv_entry["title"] = entry["properties"]["Title"]["title"][0]["plain_text"]
		if entry["properties"]["Relevance Score"].get("select") is not None:
			biorxiv_entry["relevance"] = entry["properties"]["Relevance Score"].get("select").get("name")
		else:
			biorxiv_entry["relevance"] = ""

		#biorxiv_entry["abstract"] = get_abstract(entry["id"],notionHeaders)
		biorxiv_entry["abstract"] = get_abstract_from_biorxiv(biorxiv_entry["doi"])
		#print_pretty_json(biorxiv_entry)
	return reformatted_db

def get_abstract_from_biorxiv(doi):
	url = "https://api.biorxiv.org/details/biorxiv/"+doi
	r = requests.get(url)
	print(json.dumps(r.json(),indent=2))
	abstract = r.json()["collection"][0]["abstract"]
	return abstract
#with open('db.json','r') as openfile:	json_object = json.load(openfile)
#notion_db = json_object['results']


#revised_db = translate_notion_to_bioRxiv_format(notion_db)
#print("reading page")
#page = readPage("f7412817a8a0444ba4f72828cc47ec46",notionHeaders)
#print(page)

#print(json.dumps(revised_db,indent=4))
#papers = query_bioRxiv("2022-11-28","2022-11-28")
#print(json.dumps(papers,indent=2))

#for paper in papers:
#	create_paper_in_notion(notionDatabaseId,notionHeaders,paper)



#for paper in papers:
#	createPaper(databaseId,headers,paper)

readDatabase(notionDatabaseId,notionHeaders)
#createPage(databaseId,headers)

## fine tune model using title, authors, author_corresponding, author_corresponding_institution, category, abstract that autocompletes "interest score: X" based on some representative scores.

## for each request that passes the above criteria, prompt gpt3 to see if it would be interesting. 
## include server, institution name, abstract, category
#for paper in filtered_collection:




## grab the corresponding author email and name out of the xml?