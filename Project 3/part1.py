"""The main program that runs gSpan. Two examples are provided"""
# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy
from sklearn import naive_bayes
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
    return p / (p + n)


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
        self.patterns = defaultdict(
            list)  # The patterns found in the end (as dfs codes represented by strings) with their cover (as a list of graph ids).
        self.minsup = minsup
        self.gid_subsets = subsets
        self.k = k

    # Stores k first confident patternss found that has not been pruned
    def store(self, dfs_code, gid_subsets):
        p, n = len(gid_subsets[0]), len(gid_subsets[1])
        support = p + n
        conf = p / support
        # Update the k best results
        min_score = self.get_min_confidence()
        if conf < min_score:  # if the confidence is smaller than the smallest score from the k positive confident pattern
            return
        # update patterns list with the new pattern
        self.patterns[(conf, support)].append((dfs_code, gid_subsets))
        if len(self.patterns.keys()) > self.k:
            min_score = self.get_min_confidence()
            min_sup = self.get_min_support(min_score)
            # Update the patterns so that it only keeps k best results corresponding the k best confidence scores by removing the least best score and support
            del self.patterns[(min_score, min_sup)]

    # to get the minimum confidence from the patterns
    def get_min_confidence(self):
        return min(map(lambda key: key[0], self.patterns.keys())) if len(self.patterns.keys()) >= self.k else 0

    # to get the minimum suuport corresponding to that score in the stored patterns
    def get_min_support(self, score):
        return min([k[1] for k in self.patterns.keys() if k[0] == score])

    # Prunes any pattern that is not frequent in the positive and negative class
    def prune(self, gid_subsets):
        p, n = len(gid_subsets[0]), len(gid_subsets[1])
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
        for pattern, gid_subsets in self.patterns:
            for i, gid_subset in enumerate(gid_subsets):
                matrices[i].append(self.create_fm_col(self.gid_subsets[i], gid_subset))
        return [numpy.array(matrix).transpose() for matrix in matrices]


def example1():
    """
	Runs gSpan with the specified positive and negative graphs, finds all frequent subgraphs in the positive class
	with a minimum positive support of minsup and prints them.
	"""

    args = sys.argv
    database_file_name_pos = args[1]  # First parameter: path to positive class file
    database_file_name_neg = args[2]  # Second parameter: path to negative class file
    k = int(args[3])  # Third parameter: k
    minsup = int(args[4])

    if not os.path.exists(database_file_name_pos):
        print('{} does not exist.'.format(database_file_name_pos))
        sys.exit()
    if not os.path.exists(database_file_name_neg):
        print('{} does not exist.'.format(database_file_name_neg))
        sys.exit()

    graph_database = GraphDatabase()  # Graph database object
    pos_ids = graph_database.read_graphs(
        database_file_name_pos)  # Reading positive graphs, adding them to database and getting ids
    neg_ids = graph_database.read_graphs(
        database_file_name_neg)  # Reading negative graphs, adding them to database and getting ids

    subsets = [pos_ids, neg_ids]  # The ids for the positive and negative labelled graphs in the database
    task = KFrequentPositiveGraphs(minsup, graph_database, subsets, k)  # Creating task

    gSpan(task).run()  # Running gSpan

    # task.patterns.sort(key=lambda pattern: confidence(len(pattern[-1][0]), len(pattern[-1][1])), reverse=True)

    # Printing frequent patterns along with their positive support:
    for conf, sup in task.patterns.keys():
        for pattern, gid_subsets in task.patterns[(conf, sup)]:
            print("{} {} {}".format(pattern, conf, sup))


if __name__ == '__main__':
    example1()
# example2()
