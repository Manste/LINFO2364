"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy
from sklearn import tree
from sklearn import metrics

from gspan_mining import gSpan
from gspan_mining import GraphDatabase
from collections import defaultdict

class PatternGraphs:
	"""
	This template class is used to define a task for the gSpan implementation.
	You should not modify this class but extend it to define new tasks
	"""

	def __init__(self, database):
		# A list of subsets of graph identifiers.
		# Is used to specify different groups of graphs (classes and training/test sets).
		# The gid-subsets parameter in the pruning and store function will contain for each subset, all the occurrences
		# in which the examined pattern is present.
		self.gid_subsets = []

		self.database = database  # A graphdatabase instance: contains the data for the problem.

	def store(self, dfs_code, gid_subsets):
		"""
		Code to be executed to store the pattern, if desired.
		The function will only be called for patterns that have not been pruned.
		In correlated pattern mining, we may prune based on confidence, but then check further conditions before storing.
		:param dfs_code: the dfs code of the pattern (as a string).
		:param gid_subsets: the cover (set of graph ids in which the pattern is present) for each subset in self.gid_subsets
		"""
		print("Please implement the store function in a subclass for a specific mining task!")

	def prune(self, gid_subsets):
		"""
		prune function: used by the gSpan algorithm to know if a pattern (and its children in the search tree)
		should be pruned.
		:param gid_subsets: A list of the cover of the pattern for each subset.
		:return: true if the pattern should be pruned, false otherwise.
		"""
		print("Please implement the prune function in a subclass for a specific mining task!")

def confidence(p, n):
	return p/(p+n)

class KFrequentPositiveGraphs(PatternGraphs):
	"""
	Finds the frequent (support >= minsup) subgraphs among the positive graphs.
	This class provides a method to build a feature matrix for each subset.
	"""

	def __init__(self, minsup, database, subsets, k):
		"""
		Initialize the task.
		:param minsup: the minimum positive support
		:param database: the graph database
		:param subsets: the subsets (train and/or test sets for positive and negative class) of graph ids.
		:param k:
		"""
		super().__init__(database)
		self.patterns = defaultdict(list)  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
		self.minsup = minsup
		self.gid_subsets = subsets
		self.k = k

	# Stores k first confident patterns found that has not been pruned
	def store(self, dfs_code, gid_subsets):
		p, n = len(gid_subsets[0]), len(gid_subsets[2])
		support = p+n
		pos_conf = p / support
		neg_conf = n / support
		conf = max(pos_conf, neg_conf) # to ensure the antimonotonicity
		# Update the k best results
		min_score = self.get_min_confidence()
		if conf < min_score: # if the confidence is smaller than the smallest score from the k positive confident pattern, don't store
			return
		#update patterns list with the new pattern
		self.patterns[(conf, support)].append((dfs_code, gid_subsets))
		if len(self.patterns.keys()) > self.k:
			min_sup = self.get_min_support(min_score)
			#Update the patterns so that it only keeps k best results corresponding the k best confidence scores by removing the least best score and support
			del self.patterns[(min_score, min_sup)]

	# to get the minimum confidence from the patterns
	def get_min_confidence(self):
		return min(map(lambda key: key[0], self.patterns.keys())) if len(self.patterns.keys()) >= self.k else 0

	# to get the minimum support corresponding to that score in the stored patterns
	def get_min_support(self, score):
		return min([k[1] for k in self.patterns.keys() if k[0] == score])

	# Prunes any pattern that is not frequent in the positive and negative class
	def prune(self, gid_subsets):
		p, n = len(gid_subsets[0]), len(gid_subsets[2])
		return p + n < self.minsup

	# creates a column for a feature matrix
	def create_fm_col(self, all_gids, subset_gids):
		subset_gids = set(subset_gids)
		bools = []
		for i, val in enumerate(all_gids):
			if val in subset_gids:
				bools.append(1)
			else:
				bools.append(0)
		return bools

	# return a feature matrix for each subset of examples, in which the columns correspond to patterns
	# and the rows to examples in the subset.
	def get_feature_matrices(self):
		matrices = [[] for _ in self.gid_subsets]
		for conf, sup in self.patterns.keys():
			for pattern, gid_subsets in self.patterns[(conf, sup)]:
				for i, gid_subset in enumerate(gid_subsets):
					matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
		return [numpy.array(matrix).transpose() for matrix in matrices]


def sequential_covering(minsup, database, subsets, k):
	test_predictions = {}
	# set the true classes for each graph transactions
	test_true = {gid: 1 for gid in subsets[1]}
	test_true.update({gid: -1 for gid in subsets[3]})
	# the rule list
	results = []
	# convert the subsets into list to ease the use of the subset
	subsets_copy = [list(l) for l in subsets]
	# 1. Find the highest scoring supervised pattern
	for _ in range(k):
		task = KFrequentPositiveGraphs(minsup, database, subsets_copy, 1) # find the "highest scoring" supervised pattern
		gSpan(task).run() # Launch the gspan task
		patterns = []
		for key, values in task.patterns.items():
			for v in values:
				patterns.append(v + key)
		if patterns:
			patterns.sort(key=lambda p: p[0]) # sort the lowest in the lexicographical order/ p[0] represents the dfs_code
			_, gid_subsets, _, _ = patterns[0]
			label = 1 if len(gid_subsets[2]) <= len(gid_subsets[0]) else -1
			results.append(patterns[0])
			# set predictions results on test set
			for gid in gid_subsets[1] + gid_subsets[3]:
				test_predictions[gid] = label
			# remove the transactions covered by the pattern from the data
			for index in range(len(gid_subsets)):
				for el in gid_subsets[index]:
					#print(subsets_copy[index], item, item in subsets_copy[index])
					subsets_copy[index].remove(el)
	# set the value for the rest of the patterns
	default = 1 if len(subsets_copy[2]) <= len(subsets_copy[0]) else -1
	for gid in subsets_copy[1] + subsets_copy[3]:
		test_predictions[gid] = default
	accuracy = sum([test_true[gid] == test_predictions[gid] for gid in test_true])/len(test_true)
	#print the results
	for dfs_code, _, conf, support in results:
		print("{} {} {}".format(dfs_code, conf, support))
	print([test_predictions[k] for k in test_true])
	print('accuracy: {}'.format(accuracy))
	print()  # Blank line to indicate end of fold.
	#print(patterns)


def example3():
	"""
	Runs gSpan with the specified positive and negative graphs; finds all frequent subgraphs in the training subset of
	the positive class with a minimum support of minsup.
	Uses the patterns found to train a naive bayesian classifier using Scikit-learn and evaluates its performances on
	the test set.
	Performs a k-fold cross-validation.
	"""

	args = sys.argv
	database_file_name_pos = args[1]  # First parameter: path to positive class file
	database_file_name_neg = args[2]  # Second parameter: path to negative class file
	k = int(args[3])  # Third parameter: k
	minsup = int(args[4])  # Fourth parameter: minimum support
	nfolds = int(args[5])  # Fifth parameter: number of folds to use in the k-fold cross-validation.

	if not os.path.exists(database_file_name_pos):
		print('{} does not exist.'.format(database_file_name_pos))
		sys.exit()
	if not os.path.exists(database_file_name_neg):
		print('{} does not exist.'.format(database_file_name_neg))
		sys.exit()

	graph_database = GraphDatabase()  # Graph database object
	pos_ids = graph_database.read_graphs(database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
	neg_ids = graph_database.read_graphs(database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

	# If less than two folds: using the same set as training and test set (note this is not an accurate way to evaluate the performances!)
	if nfolds < 2:
		subsets = [
			pos_ids,  # Positive training set
			pos_ids,  # Positive test set
			neg_ids,  # Negative training set
			neg_ids  # Negative test set
		]
		# Printing fold number:
		print('fold {}'.format(1))
		sequential_covering(minsup, graph_database, subsets, k)

	# Otherwise: performs k-fold cross-validation:
	else:
		pos_fold_size = len(pos_ids) // nfolds
		neg_fold_size = len(neg_ids) // nfolds
		for i in range(nfolds):
			# Use fold as test set, the others as training set for each class;
			# identify all the subsets to be maintained by the graph mining algorithm.
			subsets = [
				numpy.concatenate((pos_ids[:i * pos_fold_size], pos_ids[(i + 1) * pos_fold_size:])),  # Positive training set
				pos_ids[i * pos_fold_size:(i + 1) * pos_fold_size],  # Positive test set
				numpy.concatenate((neg_ids[:i * neg_fold_size], neg_ids[(i + 1) * neg_fold_size:])),  # Negative training set
				neg_ids[i * neg_fold_size:(i + 1) * neg_fold_size],  # Negative test set
			]
			# Printing fold number:
			print('fold {}'.format(i+1))
			sequential_covering(minsup, graph_database, subsets, k)

if __name__ == '__main__':
	example3()
