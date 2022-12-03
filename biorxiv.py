import requests
import json
from datetime import date
import re

## note the biorxiv query function only filters through the first 100 results returned -- 
## if that day has more than 100 results, it won't return them all.


# if the abstract has a paragraph with >4000 words, it'll have issues.
published_on = "2022-11-30"

today = date.today().strftime("%Y-%m-%d")

token = "secret_Tx1aAu8K56ruGDox2o1Zpg7wShZfSRvAtqjkpKbY6M1"
notionHeaders = {
	"Authorization": "Bearer " + token,
	"Content-Type": "application/json",
	"Notion-Version": "2022-06-28"
}


notionDatabaseId = "0b3a840375e04b9bbe2c2ec98729f132"


interesting_categories = ["biophysics", "synthetic biology", "cell biology","genetics", "genomics","biochemistry", "molecular biology"]
sometimes_interesting_categories = [""]




def queryBioRxiv(published_on):
	## query something like "https://api.biorxiv.org/details/biorxiv/2022-11-30/2022-11-30"
	print("querying biorxiv for the date: " + published_on)
	r = requests.get("https://api.biorxiv.org/details/biorxiv/"+published_on+"/"+published_on)
	messages = r.json()['messages']
	collection = r.json()['collection']



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
	#print(len(filtered_collection))
	#print(type(filtered_collection))
	return filtered_collection



def readDatabase(databaseId, notionHeaders):
	readUrl = f"https://api.notion.com/v1/databases/{databaseId}/query"
	print("requesting read")
	res = requests.request("POST", readUrl, headers=notionHeaders)
	print("request received")
	data = res.json()

	print(res.status_code)
	print(json.dumps(res.json(),indent=2))

	with open('./db.json', 'w', encoding='utf8') as f:
		json.dump(data, f, ensure_ascii=False)

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
def createPaperInNotion(databaseId, notionHeaders,paper):
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



papers = queryBioRxiv("2022-11-28")
#print(json.dumps(papers,indent=2))

for paper in papers:
	createPaperInNotion(notionDatabaseId,notionHeaders,paper)



#for paper in papers:
#	createPaper(databaseId,headers,paper)

#readDatabase(databaseId,headers)
#createPage(databaseId,headers)

## fine tune model using title, authors, author_corresponding, author_corresponding_institution, category, abstract that autocompletes "interest score: X" based on some representative scores.

## for each request that passes the above criteria, prompt gpt3 to see if it would be interesting. 
## include server, institution name, abstract, category
#for paper in filtered_collection:




## grab the corresponding author email and name out of the xml?