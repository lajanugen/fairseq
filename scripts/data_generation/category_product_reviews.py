import gzip
import pickle

review_path = '/home/llajan/b6/aggressive_dedup.json.gz'
meta_path ='/home/llajan/b6/metadata.json'

with open('/home/llajan/fairseq/scripts/stats/product_categories.pkl', 'rb') as f:
  product_categories = pickle.load(f)

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

with open('category_product_reviews.pkl', 'wb') as f:
  pickle.dump(category_to_reviews, f)
