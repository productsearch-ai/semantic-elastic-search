import pandas as pd
from urllib import request
import os
from elasticsearch import Elasticsearch,helpers
import json

# connection to elasticsearch

es = Elasticsearch(
    ['localhost'],
    scheme="http",
    port=9200
)

# loadiing mapping
index_mapping_file='index_mapping.json'
with open(index_mapping_file) as json_file:
    index_mapping = json.load(json_file)

# create index
print("creating 'example_index' index...")
index_name='product_search'
es.indices.create(index = index_name, body = index_mapping)
