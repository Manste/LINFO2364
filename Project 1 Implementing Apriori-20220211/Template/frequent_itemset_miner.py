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


class Dataset:
	"""Utility class to manage a dataset stored in a external file."""

	def __init__(self, filepath):
		"""reads the dataset file and initializes files"""
		self.transactions = list()
		self.items = set()

		try:
			lines = [line.strip() for line in open(filepath, "r")]
			lines = [line for line in lines if line]  # Skipping blank lines
			for line in lines:
				transaction = list(map(int, line.split(" ")))
				self.transactions.append(transaction)
				for item in transaction:
					self.items.add(item)
		except IOError as e:
			print("Unable to read dataset file!\n" + e)

	def trans_num(self):
		"""Returns the number of transactions in the dataset"""
		return len(self.transactions)

	def items_num(self):
		"""Returns the number of different items in the dataset"""
		return len(self.items)

	def get_transaction(self, i):
		"""Returns the transaction at index i as an int array"""
		return self.transactions[i]


class Candidate:
	def __init__(self, parent, itemset, frequency):
		self.parent = parent
		self.itemset = itemset
		self.frequency = frequency


def apriori(filepath, minFrequency):
	"""Runs the apriori algorithm on the specified file with the given minimum frequency"""
	dataset = Dataset(filepath)
	level = 0
	candidates = [None]
	"""
	While there always exist candidates generated 
	"""
	while len(candidates) != 0:
		# detect frequent itemset
		candidates = generate_candidates(dataset, level, candidates)
		frequencies(candidates, dataset)
		frequent_candidates = check_frequencies(candidates, minFrequency)
		level += 1


def check_frequencies(frequency_per_candidate, min_frequency):
	frequent_candidates = []
	for candidate in frequency_per_candidate:
		if candidate.frequency >= min_frequency:
			frequent_candidates.append(candidate)
			print(candidate.itemset, " ", candidate.frequency)
	return frequent_candidates


def frequencies(candidates, dataset):
	"""Counting candidates using the naive process"""
	if candidates[0].itemset is None: return dataset.trans_num()
	for candidate in candidates:
		"""For each transaction,we check if it contains the itemset."""
		for transaction in dataset.transactions:
			if candidate.itemset.issubset(transaction):
				candidate.frequency += 1/dataset.trans_num()


"""
We will generate candidate based on frequent itemset detected
"""
def generate_candidates(dataset, level, last_candidates=[]):
	from itertools import combinations
	new_candidates = []
	if level == 0:
		new_candidates = [Candidate(None, None, dataset.trans_num())]
	elif level == 1:
		for item in dataset.items:
			new_candidates.append(Candidate(last_candidates, {item}, 0))
	else:
		for itemset in combinations(last_candidates, level):
			temp_parent = list(itemset)
			temp_set = [parent.itemset for parent in last_candidates]
			temp_lst = list(frozenset().union(*temp_set))
			# Check if the corresponding union
			if len(temp_lst) == level:
				# ordonne la liste
				temp_lst.sort()
				temp_itemset = frozenset(temp_lst)
				new_candidates.append(Candidate(temp_parent, temp_itemset, 0))
		print(new_candidates)
	return new_candidates


def alternative_miner(filepath, minFrequency):
	"""Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
	# TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
	print("Not implemented")

apriori('../Datasets/toy.dat', 0.1)