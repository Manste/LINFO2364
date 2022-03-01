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

__authors__ = "Group 8: Manuelle Ndamtang <manuelle.ndamtang@student.uclouvain.be>,
						Saskia Juffern <saskia.juffern@student.uclouvain.be>"
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
			print("{}  ({})".format(sorted(list(candidate)), frequency))
	return frequent_candidates


def frequencies(candidates, dataset):
	"""Counting candidates using the naive process"""
	items_frequencies = {}
	if len(candidates) == 0 or candidates[0] is None: return dataset.trans_num()
	for candidate in candidates:
		"""For each transaction,we check if it contains the itemset."""
		for transaction in dataset.transactions:
			if candidate.issubset(transaction):
				c = frozenset(candidate)
				if c in items_frequencies.keys():
					items_frequencies[c] += 1 / dataset.trans_num()
				else:
					items_frequencies[c] = 1 / dataset.trans_num()
	return items_frequencies


"""
We will generate candidate based on frequent itemset detected
"""
def generate_candidates(dataset, level, last_candidates=[]):
	from itertools import combinations
	new_candidates = []
	if level == 0:
		new_candidates = [None]
	elif level == 1:
		for item in dataset.items:
			new_candidates.append({item})
	else:
		for itemset in combinations(last_candidates, level):
			temp_parent = list(itemset)
			lst = frozenset().union(*temp_parent)
			if len(lst) == level:
				new_candidates.append(lst)
	return new_candidates


def alternative_miner(filepath, minFrequency):
	"""Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
	# TODO: either second implementation of the apriori algorithm or implementation of the depth first search algorithm
	dataset = Dataset(filepath)
	minNProd = 2
	minSupport = 3 / dataset.trans_num()
	maxLen = max([len(dataset.get_transaction(x)) for x in range(dataset.trans_num())])

	#create dictionary of sets of transactions for every item
	tid = dict()
	for i in range (dataset.trans_num()):
		for elem in dataset.get_transaction(i):
			if not elem in tid.keys():
				tid[elem] = set()
				tid[elem].add(i)
			else:
				tid[elem].add(i)

	#Step 2: todo: filter tid dict with min support

	for idx in tid.keys():
		#recursive call
		eclatRec(idx, tid)


	#maybe better if this is done recursively
	new_tid = dict()
	for elem1 in tid.keys():
		for elem2 in tid.keys():
			if not elem1 == elem2:
				inter = tid[elem1].intersection(tid[elem2])
				support = len(inter)/dataset.trans_num()
				if support > minFrequency:
					new_tid[elem1, elem2] = inter
	return tid, new_tid

def eclatRec(focusedIdx, tid):
	for i in tid.keys():
		if not i == focusedIdx:
			intersection = tid[focusedIdx].intersection(tid[i])



## didn't yet figure out how to go from minFreq to support
di, new_tid = alternative_miner('../Datasets/toy.dat', 0.3)
print(di.keys(), new_tid.keys())


#apriori('../Datasets/toy.dat', 0.125)