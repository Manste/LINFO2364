"""
Skeleton file for the project 1 of the LINGI2364 course.
Use this as your submission file. Every piece of code that is used in your program should be put inside this file.

This file given to you as a skeleton for your implementation of the Apriori and Depth
First Search algorithms. You are not obligated to use them and are free to write any class or method as long as the
following requirements are respected:

Your apriori and alternativeMiner methods must take as parameters a string corresponding to the path to a valid
dataset file and a double corresponding to the minimum frequency.
You must write on the standard output (use the print() method) all the itemsets that are frequent in the dataset file
according to the minimum frequency given. Each itemset has to be printed on one line following the format:
[<item 1>, <item 2>, ... <item k>] (<frequency>).
Tip: you can use Arrays.toString(int[] a) to print an itemset.

The items in an itemset must be printed in lexicographical order. However, the itemsets themselves can be printed in
any order.

Do not change the signature of the apriori and alternative_miner methods as they will be called by the test script.

__authors__ = "Group 8: Manuelle Ndamtang <manuelle.ndamtang@student.uclouvain.be>"
"""
from itertools import combinations
import numpy as np

class Dataset:
	"""Utility class to manage a dataset stored in a external file."""

	def __init__(self, filepath):
		"""reads the dataset file and initializes files"""
		temp_transactions = list()
		self.items = set()

		try:
			lines = [line.strip() for line in open(filepath, "r")]
			lines = [line for line in lines if line]  # Skipping blank lines
			for line in lines:
				transaction = np.asarray(line.split(" "), dtype=np.int64)
				temp_transactions.append(transaction)
				for item in transaction:
					self.items.add(item)
		except IOError as e:
			print("Unable to read dataset file!\n" + e)
		self.transactions = np.asarray(temp_transactions, dtype=object)

	def trans_num(self):
		"""Returns the number of transactions in the dataset"""
		return len(self.transactions)

	def items_num(self):
		"""Returns the number of different items in the dataset"""
		return len(self.items)

	def get_transaction(self, i):
		"""Returns the transaction at index i as an int array"""
		return self.transactions[i]


def apriori(filepath, minFrequency):
	"""Runs the apriori algorithm on the specified file with the given minimum frequency"""
	dataset = Dataset(filepath)
	level = 1
	candidates = [None]
	frequent_itemsets_per_level = {
		0 : candidates
	}
	"""
	While there always exist candidates generated 
	"""
	while True:
		# detect frequent itemset
		candidates = generate_candidates(dataset, level, candidates)
		if not candidates:
			return
		items_frequencies = frequencies(candidates, dataset)
		frequent_itemsets_per_level[level] = check_frequencies(items_frequencies, minFrequency)
		level += 1


def check_frequencies(frequency_per_candidate, min_frequency):
	frequent_candidates = []
	for candidate, frequency in frequency_per_candidate.items():
		if frequency >= min_frequency:
			frequent_candidates.append(candidate)
			print("{}  ({})".format(candidate, frequency))
	return frequent_candidates


def frequencies(candidates, dataset):
	items_frequencies = {}
	if len(candidates) == 0 or candidates[0] is None: return dataset.trans_num()
	for candidate in candidates:
		"""For each transaction,we check if it contains the itemset."""
		filtered_transactions = dataset.transactions.apply(lambda row: items_frequencies[candidate] = 1/dataset.trans_num() if candidate in items_frequencies else items_frequencies[candidate] = items_frequencies[candidate] + 1/dataset.trans_num() )
	temp_items = np.array(items_frequencies.keys()).reshape(-1, 1)
	temp_frequencies = np.array(items_frequencies.values()).reshape(-1, 1)
	print(items_frequencies)
	return np.array((temp_items, temp_frequencies), axis=1)


"""
We will generate candidate based on frequent itemset detected
"""
def generate_candidates(dataset, level, last_candidates=[]):
	new_candidates = []
	if level == 0:
		new_candidates = [None]
	elif level == 1:
		for item in dataset.items:
			new_candidates.append([item])
	else:
		for itemset in combinations(last_candidates, level):
			temp_parent = list(itemset)
			lst = frozenset().union(*temp_parent)
			if len(lst) == level:
				new_candidates.append(list(lst))
	return new_candidates


def alternative_miner(filepath, minFrequency):
	"""Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
	# TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
	print("Not implemented")

apriori('../Datasets/toy.dat', 0.9)