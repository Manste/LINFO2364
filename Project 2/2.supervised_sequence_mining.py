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
        self.P = self.dataset1.trans_num()
        self.N = self.dataset2.trans_num()
        self.min_score = 0
        self.pos_min_support = 0
        self.results = {}
        self.covers_per_result = {}
        self.k = k
        self.k_first_scores_values = set()

    def score_per_pattern(self, patterns):
        """This function gives score per patterns"""
        support_per_pattern = {pattern: len(set().union(list(map(lambda pos: pos[0], patterns[pattern])))) for pattern in patterns}
        positive_per_pattern = {pattern: len(set(map(lambda x: x[0], filter(lambda x: x[0] < self.dataset1.trans_num(), patterns[pattern])))) for pattern in patterns}
        negative_per_pattern = {pattern: support_per_pattern[pattern] - positive_per_pattern[pattern] for pattern in patterns}
        return {pattern: (positive_per_pattern[pattern], negative_per_pattern[pattern], self.wracc(positive_per_pattern[pattern], negative_per_pattern[pattern])) for pattern in patterns}

    def wracc(self, p, n):
        return round(((self.P*self.N)/(self.P + self.N)**2) * (p/self.P - n/self.N), 5)

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
        # get the supports and score for each sequence found
        next_sequences_score = self.score_per_pattern(next_sequences)

        # keep k best results
        first_values = dict(filter(lambda x: x[1][-1] >= self.min_score, next_sequences_score.items())) # keep the values that are bigger or equals to the min_score
        self.results.update(first_values)
        self.k_first_scores_values = sorted(set(map(lambda x: x[-1], self.results.values())), reverse=True)[0:self.k]
        self.k_first_scores_values = self.k_first_scores_values[0:self.k]

        # update results with the k-first values
        results_copy = self.results.copy()
        self.results = {}
        for pattern in results_copy.keys():
            if results_copy[pattern][-1] in self.k_first_scores_values:
                self.results[pattern] = results_copy[pattern]
                if pattern not in self.covers_per_result:
                    self.covers_per_result[pattern] = next_sequences[pattern]

        # Update the minimum score which is the minimum score stored from the list of k best score
        # and update the positive minimum support which will be used in our heuristic function in order
        # to easily search through our recursion
        if len(self.k_first_scores_values) == self.k:
            self.min_score = self.k_first_scores_values[-1]# Threshold computation
            self.pos_min_support = (((self.P + self.N)**2) * self.min_score) / (self.P + self.N)

        # launch the recursive search for frequent sequences
        for next_pattern, positions in next_sequences.items():
            # remove infrequent sequences
            p, n, score = next_sequences_score[next_pattern]
            if p < self.pos_min_support:
                #print(next_pattern, next_sequences_score[next_pattern], p, self.pos_threshold, self.k_first_scores_values)
                return
            self.dfs(next_pattern, positions)

    def main(self):
        """Main function that start research of pattern through recursion"""
        pattern = tuple()
        positions = [(i, -1) for i in range(len(self.transactions))]
        self.dfs(pattern, positions)
        self.print_results()

    def print_results(self):
        """Print the results obtained"""
        for pattern, (p, n, score) in self.results.items():
            print("[{}] {} {} {}".format(', '.join(map(str, pattern)), p, n, score))

def main():
    pos_filepath = sys.argv[1] # filepath to positive class file
    neg_filepath = sys.argv[2] # filepath to negative class file
    k = int(sys.argv[3])
    prefixSpan = PrefixSpan(pos_filepath, neg_filepath, k)
    prefixSpan.main()

if __name__ == "__main__":
    main()