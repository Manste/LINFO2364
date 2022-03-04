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

from itertools import combinations
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
        items_frequencies = frequencies(candidates, dataset)
        check_frequencies(items_frequencies, minFrequency)
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
        for item in dataset.items:
            new_candidates.append({item})
    else:
        for itemset in combinations(last_candidates, level):
            temp_parent = list(itemset)
            lst = frozenset().union(*temp_parent)
            if len(lst) == level:
                new_candidates.append(lst)
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
    eclat(vertical_dataset, None, minFrequency, dataset)


"""
    This function recursivily search for the frequent itemset and stop it search for a specific
"""


def eclat(vertical_dataset, itemset, minFrequency, dataset):
    items = sorted(list(dataset.items))
    if itemset:
        frequency = len(frozenset().union(*vertical_dataset.values())) / dataset.trans_num()
        union_set = sorted(list(itemset))
        idx = items.index(union_set[-1]) + 1
        if frequency < minFrequency:
            return
        else:
            print_itemset(itemset, frequency)
    else:
        idx = 0
    for index, item in enumerate(items[idx:]):
        if itemset is None:
            item_set = {item}
        else:
            item_set = set(union_set.copy())
            item_set.add(item)
        projected_dataset = projected_database(vertical_dataset, frozenset(item_set), minFrequency, dataset.trans_num())
        eclat(projected_dataset, frozenset(item_set), minFrequency, dataset)


if __name__ == '__main__':
    from time import perf_counter
    import pandas as pd
    import tracemalloc

    apriori_frame_performance = pd.DataFrame(columns=["minFrequency", "duration", "memory"])
    alternative_miner_frame_performance = pd.DataFrame(columns=["minFrequency", "duration", "memory"])

    frames = {"apriori": apriori_frame_performance, "eclat": alternative_miner_frame_performance}
    filename = "Datasets/toy.dat"
    minFrequencies = [0.125]#[0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for minFrequency in minFrequencies:
        for key in frames.keys():
            tic = perf_counter()
            tracemalloc.start()
            apriori(filename, minFrequency)
            tracemalloc.take_snapshot()
            tracemalloc.stop()
            duration = perf_counter() - tic
            new_row = pd.DataFrame({
                "minFrequency": [key],
                "duration": [duration],
                "memory": [gamma]
            })
            frame2 = pd.concat([frame2, new_row], ignore_index=True)