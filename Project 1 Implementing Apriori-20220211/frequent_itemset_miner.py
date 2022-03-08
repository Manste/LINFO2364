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

__authors__ = "Group 8: Manuelle Ndamtang <manuelle.ndamtang@student.uclouvain.be>
"""

from collections import defaultdict


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
    """
    While there always exist candidates generated 
    """
    while True:
        """
            We detect the frequent itemset candidate using the antimocity approach for each level:
            We generate candidates using only superset that are already frequent at each level
        """
        candidates = generate_candidates(dataset, level, candidates)
        if not candidates:
            return
        items_supports = frequencies(candidates, dataset)
        candidates = check_frequencies(items_supports, minFrequency) # update candidates variables so that in the next search, it will generate candidates only based on the frequent ones
        level += 1


"""
	After getting the frequency for each itemset, we exclude the non-frequent itemset and print only 
	the frequent one.
"""
def check_frequencies(frequency_per_candidate, min_frequency):
    frequent_candidates = []
    for candidate, frequency in frequency_per_candidate.items():
        if frequency >= min_frequency:
            frequent_candidates.append(candidate)
            print_itemset(candidate, frequency)
    return frequent_candidates


"""
    This function print the results.
"""
def print_itemset(candidate, frequency):
    print("{}  ({})".format(sorted(list(candidate)), frequency))


"""
	For each candidate, we find it frequency
"""
def frequencies(candidates, dataset):
    """
        Counting candidates using the naive process:
        for each line of the dataset, check if we find the candidate
    """
    items_frequencies = {}
    if len(candidates) == 0 or candidates[0] is None: return dataset.trans_num()
    """
        For each transaction,we check if it contains the itemset.
    """
    for candidate in candidates:
        c = frozenset(candidate)
        items_frequencies[c] = len(list(filter(c.issubset, dataset.transactions))) / dataset.trans_num()
    return items_frequencies


"""
    We will generate candidate based on frequent itemset detected
"""
def generate_candidates(dataset, level, last_candidates=[]):
    new_candidates = []
    if level == 0:
        new_candidates = [None]
    elif level == 1:
        new_candidates = [{item} for item in dataset.items]
    else:
        for index, itemset1 in  enumerate(last_candidates):
            for itemset2 in last_candidates[index:]:
                if itemset1 != itemset2 and sorted(list(itemset1))[:-1] == sorted(list(itemset2))[:-1]: # check if the end is different
                    new_candidates.append(frozenset().union(*[itemset1, itemset2]))
    return new_candidates


"""
    This function transform the dataset into a vertical representation where
    each item is map to its cover
"""
def vertical_representation(dataset):
    transaction_per_item = {frozenset({i}): [] for i in dataset.items}
    transaction_per_item = defaultdict(frozenset, transaction_per_item)
    for index, transaction in enumerate(dataset.transactions):
        for item in transaction_per_item.keys():
            if item.issubset(transaction):
                transaction_per_item[item].append(index)
    return transaction_per_item


"""
    This gives the projected database for a specific itemset
"""
def projected_database(transactions_per_item, itemset, minFrequency, total_transaction):
    projection = defaultdict(frozenset, transactions_per_item.copy())
    for item in itemset:
        temp_item = frozenset({item})
        for i in transactions_per_item.keys():
            intersection = frozenset(projection[temp_item]).intersection(projection[i])
            if len(intersection) >= minFrequency * total_transaction:  # if selected item (i) in the projected database is frequent
                projection[i] = intersection
            else:
                del projection[i]  # delete non frequent item
        if item in projection:
            del projection[temp_item]  # delete item from the itemset that are present in database
    return projection


"""
    We opted for the eclat algorithm.
"""
def alternative_miner(filepath, minFrequency):
    """Runs the alternative frequent itemset mining algorithm on the specified file with the given minimum frequency"""
    dataset = Dataset(filepath)
    vertical_dataset = vertical_representation(dataset)
    for item in dataset.items:
        item_set = {item}
        projected_dataset = projected_database(vertical_dataset, frozenset(item_set), minFrequency, dataset.trans_num())
        eclat(projected_dataset, frozenset(item_set), minFrequency, dataset)


"""
    This function recursivily search for the frequent itemset and stop it search for a specific itemset is not frequent
"""
def eclat(vertical_dataset, itemset, minFrequency, dataset):
    items = sorted(list(dataset.items))
    frequency = len(frozenset().union(*vertical_dataset.values())) / dataset.trans_num()
    union_set = sorted(list(itemset))
    idx = items.index(union_set[-1]) + 1
    if frequency < minFrequency:
        return
    else:
        print_itemset(itemset, frequency)
    for index, item in enumerate(items[idx:]): # We will generate candidate that are in a sorter manner, that why we use the index variable
        if frozenset({item}) in vertical_dataset: # to make sure we will only generate candidate that might be frequent and present in ptojected dataset
            item_set = set(union_set.copy())
            item_set.add(item)
        else:
            continue
        projected_dataset = projected_database(vertical_dataset, frozenset(item_set), minFrequency, dataset.trans_num())
        eclat(projected_dataset, frozenset(item_set), minFrequency, dataset)


if __name__ == '__main__':
    from time import perf_counter
    import pandas as pd
    import tracemalloc

    filenames = {
        "mushroom.dat": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
        "accidents.dat": [0.975, 0.95, 0.925, 0.9, 0.875, 0.85, 0.825, 0.8],
        "chess.dat": [0.975, 0.95, 0.925, 0.9, 0.875, 0.85, 0.825, 0.8]
    }
    for filename in filenames:

        frames = {
            "apriori": {
                "frame": pd.DataFrame(columns=["minFrequency", "duration", "currentMemoryUsage", "Peak"]),
                "function": apriori
            }, "eclat": {
                "frame": pd.DataFrame(columns=["minFrequency", "duration", "currentMemoryUsage", "Peak"]),
                "function": alternative_miner
            }
        }
        plus_folder = './Datasets/{}'.format(filename)
        minFrequencies = filenames[filename]
        for minFrequency in minFrequencies:
            for key in frames.keys():
                print("\n\nFrequent itemsets of {} with minFrequency {} and {} algorithm".format(filename, minFrequency, key))
                # save the stats
                tic = perf_counter()
                tracemalloc.start()
                frames[key]["function"](plus_folder, minFrequency)
                current, peak = tracemalloc.get_traced_memory()
                duration = perf_counter() - tic
                new_row = pd.DataFrame({
                    "minFrequency": [minFrequency],
                    "duration": [duration],
                    "currentMemoryUsage": [current/(1024**2)],# In MB
                    "Peak": [peak/(1024**2)] # in MB
                })
                tracemalloc.stop()
                frames[key]["frame"] = pd.concat([frames[key]["frame"], new_row], ignore_index=True)
                frames[key]["frame"].to_csv("./Performance/{}{}.csv".format(key, filename[0:-4]), index=False, header=True)