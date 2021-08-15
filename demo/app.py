import pandas as pd
from urllib import request
import os
from elasticsearch import Elasticsearch,helpers
import json
from tqdm import tqdm
import pystache
from flask import render_template
from flask import Flask
from flask import request
from flask import jsonify

# loading the search template
index_name = 'product_search'
query_template_path='query_template.json'
with open(query_template_path) as f:
  query_template = json.load(f)
query_template_str=json.dumps(query_template)

es = Elasticsearch(
    ['localhost'],
    scheme="http",
    port=9200
)

def construct_query(keywords, pageno,pagesize):
    page_from=(pageno-1)*pagesize
    parameters={'start':page_from,'size':pagesize,'keywords':keywords}
    query = pystache.render(query_template_str, parameters)
    return query

def search(keywords, pageno,index_name, es_instance,pagesize=20):
    search_query=construct_query(keywords, pageno,pagesize)
    results=es_instance.search(index=index_name,body=search_query)
    return results

app = Flask(__name__,
            static_url_path='',
            static_folder='web/static',
            template_folder='web/templates')
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

@app.route('/',methods = ['GET'])
def index():
    results = {}
    if request.method == 'GET':
        keyword=request.args.get('keyword')
        if keyword==None:
            keyword=''
        else:
            results=search(keyword, 1,index_name, es)

        return render_template('index.html',keyword=keyword, results=results, title='Search demo')
