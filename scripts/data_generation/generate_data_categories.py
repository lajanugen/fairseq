import pickle

def writeData(data, fname):
  train_fh = open(fname, 'w')

  for task in data:
    if len(task) == 0:
      train_fh.write("\n")
    else:
      train_fh.write("\n".join(task))
      train_fh.write("\n\n")

  train_fh.close()

def read_categories_file(fname):
  categories = []
  with open(fname, 'r') as f:
    for line in f.readlines():
      category = ' '.join(line.split()[:-1])
      categories.append(category)
  return categories

categories_dir = 'train_test_categories'
data_dir = '/tmp/data'
category_product_reviews_file = 'category_product_reviews.pkl'

train_categories = read_categories_file('%s/train_categories.txt' % categories_dir)
test_categories = read_categories_file('%s/test_categories.txt' % categories_dir)

with open(category_product_reviews_file, 'rb') as f:
  category_reviews = pickle.load(f)

max_products_per_cat = 2
max_reviews_per_product = 5
min_reviews_per_product = 3

num_data = 0

training_tasks = []
training_tasks_valid = []
valid_tasks = []

for category, reviews in category_reviews.items():

  parent_category = category.split('.')[0]
  category_data = category_reviews[category]
  num_products_per_cat = 0
  if parent_category in train_categories:
    for product in category_data:
      if num_products_per_cat >= max_products_per_cat:
        product_reviews = category_data[product][:max_reviews_per_product]
        if len(product_reviews) >= min_reviews_per_product:
          valid_tasks.append(product_reviews)
        break
      product_reviews = category_data[product][:max_reviews_per_product]
      if len(product_reviews) >= min_reviews_per_product:
        num_products_per_cat += 1
        num_data += len(product_reviews)
        training_tasks.append(product_reviews)
        val_data = category_data[product][max_reviews_per_product:2*max_reviews_per_product]
        training_tasks_valid.append(val_data)
      print(num_data)

assert len(training_tasks) == len(training_tasks_valid)

writeData(training_tasks, '%s/train.train' % data_dir)
writeData(training_tasks_valid, '%s/train.train.valid' % data_dir)
writeData(valid_tasks, '%s/train.valid' % data_dir)


max_products_per_cat = 1
max_reviews_per_product = 20
min_reviews_per_product = 10
num_train_reviews = 5

training_tasks = []
valid_tasks = []

for category, reviews in category_reviews.items():

  parent_category = category.split('.')[0]
  category_data = category_reviews[category]
  num_products_per_cat = 0
  if parent_category in test_categories:
    for product in category_data:
      if num_products_per_cat >= max_products_per_cat:
        break
      product_reviews = category_data[product][:max_reviews_per_product]
      if len(product_reviews) >= min_reviews_per_product:
        num_products_per_cat += 1
        training_tasks.append(product_reviews[:num_train_reviews])
        valid_tasks.append(product_reviews[num_train_reviews:])

print('Num test tasks: %d' % len(training_tasks))

writeData(training_tasks, '%s/test.train' % data_dir)
writeData(valid_tasks, '%s/test.valid' % data_dir)
