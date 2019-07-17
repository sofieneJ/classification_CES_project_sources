from smart_open import smart_open
import json
from gensim.scripts import segment_wiki 

# iterate over the plain text data we just created
for line in smart_open('frwiki-latest-abstract2.xml.gz'):
    # decode each JSON line into a Python dictionary object
    article = json.loads(line)
    # each article has a "title", a mapping of interlinks and a list of "section_titles" and "section_texts".
    print("Article title: %s" % article['title'])
    print("Interlinks: %s" + article['interlinks'])
    for section_title, section_text in zip(article['section_titles'], article['section_texts']):
        print("Section title: %s" % section_title)
        print("Section text: %s" % section_text)

# mytext = segment_wiki.extract_page_xmls('frwiki-latest-abstract2.xml')
# print (mytext.__next__())