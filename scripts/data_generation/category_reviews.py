import gzip
import pickle

review_path = '/home/llajan/b6/aggressive_dedup.json.gz'
meta_path ='/home/llajan/b6/metadata.json'

ct = 0
product_categories = {}
with open(meta_path, 'r') as f:
  line = f.readline()
  while line:
    ct += 1
    if ct % 10000 == 0:
      print(ct)
    line = line.strip()
    d = eval(line)
    if 'categories' in d and len(d['categories']) == 1:
      category = d['categories'][0]
      if category:
        category = '.'.join(category)
        if category not in product_categories:
          product_categories[category] = set()
        product_categories[category].add(d['asin'])
    line = f.readline()

with open('stats/product_categories.pkl', 'wb') as f:
  pickle.dump(product_categories, f)
