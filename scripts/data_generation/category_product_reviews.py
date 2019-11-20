import gzip
import pickle

review_path = '/mnt/brain4/datasets/amazon_reviews/aggressive_dedup.json.gz'
meta_path ='/mnt/brain4/datasets/amazon_reviews/metadata.json'

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

product_to_category = {}
for category in product_categories:
  for product in product_categories[category]:
    product_to_category[product] = category

num_categories = len(product_categories.keys())
done_count = 0
N = 10

category_to_reviews = {}

ct = 0
g = gzip.open(review_path, 'r')
for l in g:
  ct += 1
  if ct % 10000 == 0:
    print(ct)
  d = eval(l)
  key = d['asin'] 
  value = d['reviewText']

  if key in product_to_category:
    category_str = product_to_category[key]
    if category_str:
      category_str = category_str.replace('/', '_')
      if category_str not in category_to_reviews:
        category_to_reviews[category_str] = {}
      if key not in category_to_reviews[category_str]:
        category_to_reviews[category_str][key] = []
      category_to_reviews[category_str][key].append(value)

with open('/tmp/category_product_reviews.pkl', 'wb') as f:
  pickle.dump(category_to_reviews, f)
