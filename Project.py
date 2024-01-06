import Toolkit
import pandas as pd
import numpy as np
import math
import pickle
import cProfile

class Node:
   def __init__(self, attribute=None, label=None, branches=None, split_value = None, error = None):
      self.attribute = attribute    # the attribute to split on
      self.label = label            # the class label (if leaf node)
      self.branches = branches      # a dictionary of child nodes
      self.split_value = split_value # the split value, if attribtue is continuous
      self.error = error
# ----- Used for classification trees -----
def gratio(f, numeric = False):
   iv = IV(f)
   if not iv:
      return 0
   return (H(f) - E(f, numeric)) / iv

def H(D):
   k = D['class'].unique()
   magnitude = len(D)
   ret = 0
   for l in k:
      ret += -1 * len(D[D['class'] == l]) / magnitude * math.log2(len(D[D['class'] == l]) / magnitude)
   return ret

def E(f, numeric):
   m = f[f.columns[0]].unique()
   index = 0
   magnitude = len(f) 
   if numeric:
      index = 1
      m = f[f.columns[1]].unique()
   ret = 0
   for j in m:
      ret += len(f[f[f.columns[index]] == j]) / magnitude * H(f[f[f.columns[index]] == j])
   return ret

def IV(f):
   m = f[f.columns[0]].unique()
   magnitude = len(f) 
   ret = 0
   for j in m:
      N = len(f[f[f.columns[0]] == j])
      if N != 0:
         ret += -1 * N / magnitude * math.log2(N / magnitude)
   return ret
# ------------------------------------------
# ----- Used for regression trees ----------
def square_error(f, split):
   total = 0
   #if split:
   subset = f[f[f.columns[0]] <= split]
   for l in range(len(subset)):
         total += (subset.iloc[l]['class'] - subset['class'].mean())**2
   subset = f[f[f.columns[0]] > split]
   for l in range(len(subset)):
         total += (subset.iloc[l]['class'] - subset['class'].mean())**2
   #else:
   #   for j in f[f.columns[0]].unique():
   #      for l in range(len(f)):
   #         total += (f.iloc[l]['class'] - f['class'].mean())**2 * (1 if f[l])

   return total / len(f)
# ------------------------------------------
def node_entropy(x):
   entropy = 0
   total_instances = len(x)

   for class_label in x["class"].unique():
      class_instances = x[x["class"] == class_label]
      class_probability = len(class_instances) / total_instances
      class_entropy = - class_probability * math.log2(class_probability)
      entropy += class_entropy
   return entropy

def generate_leaf(x):
   leaf = pd.Series(index = list(x["class"])).to_dict()
   for key, val in leaf.items():
      leaf[key] = len(x[x['class'] == key]) / len(x)
   node = Node(label = leaf)
   return node

def generate_regression_leaf(x):
   return Node(label = x['class'].mean())

def generate_tree(x, theta, num_rows, classification, pruning):
   node = Node()
   if classification:
      if pruning:
         if node_entropy(x) < theta:
            node = generate_leaf(x)
            return node
      else:
         if node_entropy(x) == 0:
            node = generate_leaf(x)
            return node
   else:
      if pruning:
         if len(x) <= num_rows:
            node = generate_regression_leaf(x)
            return node
      else:
         if len(x) <= num_rows:
            node = generate_regression_leaf(x)
            return node
   
   i, split = split_attribute(x, classification)
   node.attribute = x.columns[i]
   branches = {}
   if split:
      node.split_value = split
      branches[0] = generate_tree(x[x[x.columns[i]] <= split], theta, num_rows, classification, pruning)
      branches[1] = generate_tree(x[x[x.columns[i]] > split], theta, num_rows, classification, pruning)
   else:
      m = x[x.columns[i]].unique()
      for j in m:
         branches[j] = generate_tree(x[x[x.columns[i]] == j], theta, num_rows, classification, pruning)
   node.branches = branches

   return node

def split_attribute(x, classification):
   cols = list(x.columns)
   max_gratio = 0
   min_sq_err = math.inf
   best_s = 0
   bestf = -1
   for i in cols:
      if i != 'class':
         if x[i].dtype.name != 'category':
            if classification:
               splits = find_splits(x[[i, 'class']])
               for s in splits:
                  temp = x.copy()
                  temp['split'] = np.where(x[i] >= s, 1, 0)
                  current_gratio = square_error(temp[['split', i, 'class']], s)
                  if current_gratio > max_gratio:
                     max_gratio = current_gratio
                     bestf = cols.index(i)
                     best_s = s
            else:
               x = x.sort_values(i)
               splits = [x[i].mean()]
               #if len(x) >= 3:
                #  splits.extend([x[i].iloc[int(len(x)/2+ 1)], x[i].iloc[int(len(x)/2-1)]])
               for s in splits:
                  current_sq_err = square_error(x[[i, 'class']], s)
                  if current_sq_err < min_sq_err:
                     min_sq_err = current_sq_err
                     bestf = cols.index(i)
                     best_s = s
         else:
            current_gratio = gratio(x[[i, 'class']])
            if current_gratio > max_gratio:
               max_gratio = current_gratio
               bestf = cols.index(i)
               best_s = 0
           
   return bestf, best_s

def find_splits(f):
   f = f.sort_values(f.columns[0])
   split_points = []
   for i in range(len(f)-1):
      if f.iloc[i]['class'] != f.iloc[i+1]['class']:
         split_points.append((f.iloc[i][f.columns[0]] + f.iloc[i + 1][f.columns[0]]) / 2)
   return set(split_points)

def predict_classification(tree, row):
   while tree.branches:
      if tree.split_value:
         if row[tree.attribute] <= tree.split_value:
            tree = tree.branches[0]
         else:
            tree = tree.branches[1]
      else:
         if row[tree.attribute] not in tree.branches:
            node = generate_leaf(pd.DataFrame(row).T)
            tree.branches[row[tree.attribute]] = node
         tree = tree.branches[row[tree.attribute]]
   ret = max(tree.label, key=tree.label.get)
   return ret

def predict_regression(tree, row):
   while tree.branches:
      if tree.split_value:
         if row[tree.attribute] <= tree.split_value:
            tree = tree.branches[0]
         else:
            tree = tree.branches[1]
      else:
         if row[tree.attribute] not in tree.branches:
            node = generate_leaf(pd.DataFrame(row).T)
            tree.branches[row[tree.attribute]] = node
         tree = tree.branches[row[tree.attribute]]
   ret = (row['class'] - tree.label)**2
   return ret

def prune(node, samples, classification):
   """ Performs pruning of the tree
   
   Recursively prunes the tree given by node, using samples from samples
   
   Args:
      node (Node): the root of the tree
      samples (dataframe): dataframe of samples that end up in
   
   Returns:
      ret (dictionary): a dictionary of run information for tuning
   """
   total = 0
   if not len(samples):
      node.error = 0
      return node

   if classification:
      for i in range(len(samples)):
         total += (1 if samples.iloc[i]['class'] != samples['class'].mode()[0] else 0)

   else:
      for i in range(len(samples)):
         total += (samples.iloc[i]['class'] - samples['class'].mean())**2
   node.error = total / len(samples)
   if not node.branches:
      return node
   
   children_error = 0
   if node.split_value:
      children_error = prune(node.branches[0], samples[samples[node.attribute] <= node.split_value], classification).error + prune(node.branches[1], samples[samples[node.attribute] > node.split_value], classification).error
   else:
      for key, branch in node.branches.items():
         children_error += prune(branch, samples[samples[node.attribute] == key], classification).error

   if node.error <= children_error:
      node.branches = {}
      node.label = (samples['class'].mode() if classification else samples['class'].mean())
      return node
   else:
      node.error = children_error
      return node


def cross_validation(df, filename, classification, pruning):
   """ Performs 5x2 cross-validation on the source dataframe
   
   The main workhorse of the program
   
   Args:
      df (dataframe): the pre-processed dataframe to be analyzed
      classification (boolean): 1 for classification task, 0 for regression
      categorical_flags (array): array of indices to indicate categorical columns
      numerical_flags (array): array of indices to indicate numerical columns
   
   Returns:
      ret (dictionary): a dictionary of run information for tuning
   """
   theta = .1
   pruned_trees = []
   trees = []
   pruned_ret = []
   ret = []
   validation = df.groupby('class', group_keys = False).apply(lambda g: g.sample(frac=0.2))
   for i in range(5):
      training = df.loc[~df.index.isin(validation.index)]
      test = training.groupby('class', group_keys = False).apply(lambda g: g.sample(frac = .5))
      training = training.loc[~training.index.isin(test.index)].reset_index(drop=True)
      test = test.reset_index(drop = True)
      validation = validation.reset_index(drop = True)

      #tree = generate_tree(training, theta, 10, classification, 0)
      trees.append(generate_tree(training, theta, 10, classification, 0))
      #path = ""
      #path = filename + str(i) 
      #if pruning:
      #   path += "_pruned" 
      #path += ".pkl"

      #with open(path, 'wb') as fp:
      #   pickle.dump(trees[i], fp)

      pruned_predictions = []
      predictions = []
      if classification:
         for j in range(len(test)):
            predictions.append(predict_classification(trees[i], test.iloc[j]))

         correct = 0
         for j in range(len(test)):
            if predictions[j] == test.iloc[j]['class']:
               correct += 1
         percent_correct = correct/len(test)
         ret.append(percent_correct)

        
      else:
         for j in range(len(test)):
            predictions.append(predict_regression(trees[i], test.iloc[j]))
         ret.append(sum(predictions)/len(predictions))

      pruned_trees.append(prune(trees[i], validation, classification))
      if classification:
         for j in range(len(test)):
            pruned_predictions.append(predict_classification(pruned_trees[i], test.iloc[j]))

         correct = 0
         for j in range(len(test)):
            if pruned_predictions[j] == test.iloc[j]['class']:
               correct += 1
         percent_correct = correct/len(test)
         pruned_ret.append(percent_correct)
      else:
         for j in range(len(test)):
            pruned_predictions.append(predict_regression(pruned_trees[i], test.iloc[j]))
         pruned_ret.append(sum(pruned_predictions)/len(pruned_predictions))

   return pruned_ret, pruned_trees, ret, trees

#dataset = 1
#data = data = Toolkit.load_from_csv(dataset)
#if dataset == 1:
#   data = data.drop(columns = ['id'])
#pruned_percent_correct, pruned_trees, percent_correct, trees = cross_validation(data, "breast_cancer", 1, 1)

#dataset = 2
#data = Toolkit.load_from_csv(dataset)
#pruned_percent_correct, pruned_trees, percent_correct, trees = cross_validation(data, "cars", 1, 0)

#dataset = 3
#data = pd.read_csv("C:/Users/brend/Documents/Classes/Machine Learning/DataSets/Congress/house-votes-84.data", header = None)
#data.columns = ["class", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid", 
#                "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback",
#                "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"]
#data = Toolkit.ordinal_conversion(data, [0])
#for col in data.columns:
#   data[col] = data[col].astype('category')
#pruned_percent_correct, pruned_trees, percent_correct, trees = cross_validation(data, "house_votes", 1, 0)

#dataset = 4
#data = Toolkit.load_from_csv(dataset)
#pruned_percent_correct, pruned_trees, percent_correct, trees = cross_validation(data, "abalone", 0, 1)

dataset = 5
data = Toolkit.load_from_csv(dataset)
pruned_percent_correct, pruned_trees, percent_correct, trees = cross_validation(data, "machine", 0, 1)

dataset = 6
data = Toolkit.load_from_csv(dataset)
pruned_percent_correct, pruned_trees, percent_correct, trees = cross_validation(data, "forest_fires", 0, 1)
print(data)