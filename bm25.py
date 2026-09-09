import math
import re
from collections import Counter


class BM25:
    """
    Simple BM25 implementation.

    Used for lexical keyword-based retrieval.
    """

    def __init__(
        self,
        documents,
        k1=1.5,
        b=0.75,
    ):
        self.documents = documents
        self.k1 = k1
        self.b = b

        self.tokenized_documents = [
            self.tokenize(doc)
            for doc in documents
        ]

        self.doc_lengths = [
            len(tokens)
            for tokens in self.tokenized_documents
        ]

        self.avgdl = (
            sum(self.doc_lengths)
            / len(self.doc_lengths)
            if self.doc_lengths
            else 0
        )

        self.term_frequencies = [
            Counter(tokens)
            for tokens in self.tokenized_documents
        ]

        self.document_frequency = Counter()

        for tokens in self.tokenized_documents:
            for term in set(tokens):
                self.document_frequency[term] += 1

        self.num_documents = len(documents)

    @staticmethod
    def tokenize(text):
        if not text:
            return []

        text = text.lower()

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text,
        )

    def score(self, query, index):
        """
        Calculate BM25 score for one document.
        """

        if (
            index < 0
            or index >= self.num_documents
        ):
            return 0.0

        query_terms = self.tokenize(query)

        if not query_terms:
            return 0.0

        frequencies = self.term_frequencies[index]
        document_length = self.doc_lengths[index]

        score = 0.0

        for term in query_terms:

            if term not in frequencies:
                continue

            tf = frequencies[term]

            df = self.document_frequency.get(
                term,
                0,
            )

            idf = math.log(
                1
                + (
                    self.num_documents
                    - df
                    + 0.5
                )
                / (
                    df
                    + 0.5
                )
            )

            denominator = (
                tf
                + self.k1
                * (
                    1
                    - self.b
                    + self.b
                    * (
                        document_length
                        / self.avgdl
                        if self.avgdl > 0
                        else 0
                    )
                )
            )

            score += (
                idf
                * (
                    tf
                    * (self.k1 + 1)
                )
                / denominator
            )

        return score

    def search(
        self,
        query,
        top_k=5,
    ):
        """
        Return:

            [
                (bm25_score, document_index),
                ...
            ]
        """

        results = []

        for index in range(
            self.num_documents
        ):
            score = self.score(
                query,
                index,
            )

            if score > 0:
                results.append(
                    (
                        score,
                        index,
                    )
                )

        results.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return results[:top_k]