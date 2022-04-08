#!/usr/bin/env python3

import sys
from collections import defaultdict
import math

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


class CloSpan:

    def __init__(self, positive, negative, k):
        self.dataset1 = Dataset(positive)
        self.dataset2 = Dataset(negative)
        self.transactions = self.dataset1.transactions + self.dataset2.transactions
        self.items = sorted(set(self.dataset1.items + self.dataset2.items))
        self.P = self.dataset1.trans_num()
        self.N = self.dataset2.trans_num()
        self.min_score = 0
        self.min_support = 0
        self.max_support = 0
        self.results = {}
        self.support_per_sequence = {}
        self.k = k
        self.k_first_scores_values = set()

    def score_per_pattern(self, patterns, score_type):
        """This function gives score per patterns"""
        support_per_pattern = {pattern: len(set().union(list(map(lambda pos: pos[0], patterns[pattern])))) for pattern in patterns}
        positive_per_pattern = {pattern: len(set(map(lambda x: x[0], filter(lambda x: x[0] < self.dataset1.trans_num(), patterns[pattern])))) for pattern in patterns}
        negative_per_pattern = {pattern: support_per_pattern[pattern] - positive_per_pattern[pattern] for pattern in patterns}
        return {pattern: (positive_per_pattern[pattern], negative_per_pattern[pattern], self.score(positive_per_pattern[pattern], negative_per_pattern[pattern], score_type)) for pattern in patterns}

    def score(self, p, n, score_type):
        if score_type == "wracc":
            return round(((self.P*self.N)/(self.P + self.N)**2) * (p/self.P - n/self.N), 5)
        if score_type == "abs-wracc":
            return abs(round(((self.P * self.N) / (self.P + self.N) ** 2) * (p / self.P - n / self.N), 5))
        if score_type == "info-gain":
            entropy = lambda x: -x * math.log2(x) - (1 - x) * math.log2(1 - x) if 1 > x > 0 else 0
            score = 0
            if self.P + self.N and p + n and self.P + self.N - p - n:
                score = entropy(self.P/(self.P + self.N)) \
                        - (p + n)/(self.P + self.N) * entropy(p / (p + n)) \
                        - (self.P + self.N - p - n) / (self.P + self.N) * entropy((self.P - p) / (self.P + self.N - p - n))
            return round(score, 5)

    # Used to cut the search tree if any pattern has a supersequence with the same positive
    # and negative support
    def cut_search_tree(self, pattern, pos, neg):
        for pat, (p, n, _) in self.support_per_sequence.items():
            if self.contains(pattern, pat) and p == pos and n == neg:
                return True
        return False

    # Used in the post-processing step in order to detect
    # Patterns which are not closed, to eliminate them
    def is_closed(self, pattern, supports):
        pos, neg, _ = supports
        for pat, (p, n, _) in self.results.items():
            if self.contains(pattern, pat) and p == pos and n == neg and len(pattern) < len(pat):
                return False
        return True

    # check if the pattern2 contains pattern1
    def contains(self, pattern1, pattern2):
        idx = 0
        for i in range(len(pattern2)):
            if pattern2[i] == pattern1[idx]: idx += 1
            if idx == len(pattern1): return True
        return False

    def clospan_score(self, positions):
        pos_score = sum([len(self.transactions[t]) - p + 1 for (t, p) in positions if t < self.dataset1.trans_num()])
        neg_score = sum([len(self.transactions[t]) - p + 1 for (t, p) in positions if t >= self.dataset1.trans_num()])
        return pos_score, neg_score

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

    def dfs(self, pattern, positions, score_type="info-gain"):
        """This function recursively searchs the patterns"""

        # get sequences and the list of positions for each transaction
        next_sequences = self.get_next_sequences(pattern, positions)
        # get the supports and score for each sequence found
        next_sequences_score = self.score_per_pattern(next_sequences, score_type)
        # update support_per_sequence
        self.support_per_sequence.update(next_sequences_score)

        # keep k best results that are closed
        first_values = dict(filter(lambda x: x[1][-1] >= self.min_score, next_sequences_score.items())) # keep the values that are bigger or equals to the min_score
        self.results.update(first_values)
        self.k_first_scores_values = sorted(set(map(lambda x: x[-1], self.results.values())), reverse=True)[0:self.k]
        self.k_first_scores_values = self.k_first_scores_values[0:self.k]

        # update results with the k-first values
        results_copy = {}
        for pattern in self.results.keys():
            if self.results[pattern][-1] in self.k_first_scores_values:
                results_copy[pattern] = self.results[pattern]
        self.results = results_copy

        # Update the minimum score which is the minimum score stored from the list of k best scores
        # and update the positive minimum support which will be used in our heuristic function in order
        # to easily search through our recursion
        if len(self.k_first_scores_values) == self.k:
            self.min_score = self.k_first_scores_values[-1]
            self.min_support = (self.P * ((self.P + self.N)**2) * self.min_score) / (self.P * self.N)
            if score_type == 'abs-wracc':
                self.max_support = (self.N * ((self.P + self.N)**2) * self.min_score) / (self.P * self.N)

        # launch the recursive search for frequent sequences
        for next_pattern, positions in next_sequences.items():
            p, n, score = next_sequences_score[next_pattern]
            if (p < self.min_support and score_type == "wracc") or ((p < self.min_support and n < self.max_support) and score_type == "abs-wracc"): # heuristic condition
                continue
            # Cut the search tree if there are some sequence having the same clospan score
            pos_score, neg_score = self.clospan_score(positions)
            if self.cut_search_tree(next_pattern, pos_score, neg_score):
                continue
            self.dfs(next_pattern, positions)

    def main(self):
        """Main function that start research of pattern through recursion"""
        pattern = tuple()
        positions = [(i, -1) for i in range(len(self.transactions))]
        self.dfs(pattern, positions)
        # Post-processing step
        self.results = dict(filter(lambda x: self.is_closed(x[0], x[1]), self.results.items()))
        self.print_results()

    def print_results(self):
        """Print the results obtained"""
        for pattern, (p, n, score) in self.results.items():
            print("[{}] {} {} {}".format(', '.join(map(str, pattern)), p, n, score))

def main():
    pos_filepath = sys.argv[1] # filepath to positive class file
    neg_filepath = sys.argv[2] # filepath to negative class file
    k = int(sys.argv[3])
    prefixSpan = CloSpan(pos_filepath, neg_filepath, k)
    prefixSpan.main()

if __name__ == "__main__":
    main()