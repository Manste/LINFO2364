#!/usr/bin/env python3

import sys
from collections import defaultdict

class Dataset:
    """This function is inspired from the Utility class to manage a dataset stored in a external file.
    given during the previous project
    """

    def __init__(self, filepath):
        """reads the dataset file and initializes files"""
        self.transactions = list()
        self.items = set()

        try:
            lines = [line.strip() for line in open(filepath, "r")]
            transaction = []
            for tid, line in enumerate(lines):
                if line:
                    item, position = line.split(' ')
                    transaction.append(item)
                    self.items.add(item)
                elif transaction:
                    self.transactions.append(transaction)
                    transaction = []
            self.items = sorted(list(self.items))
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


class PrefixSpan:

    def __init__(self, positive, negative, k):
        self.dataset1 = Dataset(positive)
        self.dataset2 = Dataset(negative)
        self.transactions = self.dataset1.transactions + self.dataset2.transactions
        self.items = sorted(set(self.dataset1.items + self.dataset2.items))
        self.min_support = 0
        self.results = {}
        self.covers_per_result = {}
        self.k = k
        self.k_first_values = set()

    def support_per_pattern(self, patterns):
        """This function gives support per patterns"""
        return {pattern: len(set().union(list(map(lambda pos: pos[0], patterns[pattern])))) for pattern in patterns}

    def get_next_sequences(self, pattern, positions):
        """"This function gets the next sequence according to the previous pattern its first positions in the
        transections """
        next_sequences = defaultdict(list)
        for tid, pid in positions:
            pid += 1
            while pid < len(self.transactions[tid]):
                new_pattern = pattern + tuple(self.transactions[tid][pid])
                next_sequences[new_pattern].append((tid, pid))
                pid += 1
        # to get the frequent sequence
        return next_sequences

    def dfs(self, pattern, positions):
        """This function recursively searchs the patterns"""

        # get sequences and the list of positions for each transaction
        next_sequences = self.get_next_sequences(pattern, positions)
        # get the support for each sequence found
        next_sequences_support = self.support_per_pattern(next_sequences)

        # keep k best results
        first_values = dict(filter(lambda x: x[1] >= self.min_support, next_sequences_support.items())) # keep the values that are bigger or equals to the min_support
        self.results.update(first_values)
        self.k_first_values = sorted(set(self.results.values()), reverse=True)[0:self.k]
        self.k_first_values = self.k_first_values[0:self.k]

        # update results with the k-first values
        results_copy = self.results.copy()
        self.results = {}
        for pattern in results_copy.keys():
            if results_copy[pattern] in self.k_first_values:
                self.results[pattern] = results_copy[pattern]
                if pattern not in self.covers_per_result:
                    self.covers_per_result[pattern] = next_sequences[pattern]

        # Update the minimum support which is the minimum support stored from the
        if len(self.k_first_values) == self.k:
            self.min_support = self.k_first_values[-1]

        # launch the recursive search for frequent sequences
        for next_pattern, positions in next_sequences.items():
            # remove infrequent sequences
            if next_sequences_support[next_pattern] < self.min_support:
                return
            self.dfs(next_pattern, positions)

    def get_frequent_symbol(self, counter_per_sequence):
        """Get the frequent pattern according to their support"""
        return [pattern for pattern, support in counter_per_sequence.items() if support > self.min_support]

    def main(self):
        """Main function that start research of pattern through recursion"""
        pattern = tuple()
        positions = [(i, -1) for i in range(len(self.transactions))]
        self.dfs(pattern, positions)
        self.print_results()

    def print_results(self):
        """Print the results obtained"""
        for pattern, support in self.results.items():
            p = len(set(map(lambda x: x[0], filter(lambda x: x[0] < self.dataset1.trans_num(), self.covers_per_result[pattern])))) # get the positive support
            n = support - p
            print("[{}] {} {} {}".format(', '.join(map(str, pattern)), p, n, support))

def main():
    pos_filepath = sys.argv[1] # filepath to positive class file
    neg_filepath = sys.argv[2] # filepath to negative class file
    k = int(sys.argv[3])
    prefixSpan = PrefixSpan(pos_filepath, neg_filepath, k)
    prefixSpan.main()

if __name__ == "__main__":
    main()