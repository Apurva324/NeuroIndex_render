"""
Populate NeuroIndex with a larger evaluation corpus.

Existing documents are preserved.
Documents with an existing title are skipped.

Requires:
    - Ollama running
    - Your existing ollama_client.py
    - Your existing storage.py / vectordb.py
"""

import config
from NeuroIndex.client import OllamaClient
from vectordb import DocumentDB


# ---------------------------------------------------------
# EVALUATION DOCUMENTS
# ---------------------------------------------------------

DOCUMENTS = [
    # =====================================================
    # CS / ALGORITHMS
    # =====================================================

    {
        "title": "Binary Search Tree",
        "text": """
A binary search tree is a tree data structure where each node
has at most two children. Values smaller than a node are placed
in its left subtree, while larger values are placed in its
right subtree. Searching, insertion, and deletion can take
O(log n) time in a balanced binary search tree, although they
can degrade to O(n) when the tree becomes highly unbalanced.
An inorder traversal of a binary search tree produces values
in sorted order.
""",
    },

    {
        "title": "Linked Lists",
        "text": """
A linked list is a linear data structure made of nodes.
Each node stores data and a reference to another node.
A singly linked list contains a pointer to the next node,
while a doubly linked list contains pointers to both the
next and previous nodes. Linked lists allow efficient
insertion and deletion when the position is known, but
accessing an arbitrary element requires traversal from
the beginning and therefore takes O(n) time.
""",
    },

    {
        "title": "Graph BFS and DFS",
        "text": """
Breadth-first search and depth-first search are fundamental
graph traversal algorithms. BFS explores vertices level by
level using a queue and is useful for finding shortest paths
in unweighted graphs. DFS explores as far as possible along
one branch before backtracking and can be implemented using
recursion or a stack. Both algorithms generally have O(V + E)
time complexity for a graph with V vertices and E edges.
""",
    },

    {
        "title": "Dynamic Programming",
        "text": """
Dynamic programming solves problems by breaking them into
overlapping subproblems and storing solutions so they do not
need to be recomputed. Two common approaches are memoization,
which stores results during recursive computation, and
tabulation, which builds a table iteratively. Dynamic
programming is useful for problems such as shortest paths,
knapsack, sequence alignment, and Fibonacci computation.
""",
    },

    {
        "title": "Sorting Algorithms",
        "text": """
Sorting algorithms arrange data according to a chosen order.
Common algorithms include bubble sort, insertion sort,
merge sort, quicksort, and heap sort. Merge sort guarantees
O(n log n) time and requires additional memory. Quicksort has
an average complexity of O(n log n), although its worst case
can be O(n squared). Choosing a sorting algorithm depends on
the data, memory constraints, and stability requirements.
""",
    },

    # =====================================================
    # AI / MACHINE LEARNING
    # =====================================================

    {
        "title": "Supervised Learning",
        "text": """
Supervised learning trains a machine learning model using
labeled examples. Each training example contains input
features and a known target output. The model learns a
mapping between inputs and targets and uses that mapping
to make predictions on unseen data. Classification predicts
discrete labels, while regression predicts continuous values.
Common supervised algorithms include logistic regression,
decision trees, support vector machines, and neural networks.
""",
    },

    {
        "title": "Unsupervised Learning",
        "text": """
Unsupervised learning works with data that does not have
explicit target labels. The goal is often to discover hidden
patterns, groups, or representations in the data. Clustering,
dimensionality reduction, and association rule learning are
common examples. Algorithms such as k-means can group similar
data points, while principal component analysis can reduce
the number of dimensions while preserving important variation.
""",
    },

    {
        "title": "Logistic Regression",
        "text": """
Logistic regression is a supervised learning algorithm commonly
used for binary classification. It applies the logistic or
sigmoid function to a weighted combination of input features
to produce a probability between zero and one. A threshold can
then convert the probability into a class prediction. Logistic
regression is simple, interpretable, and often serves as a
strong baseline for classification problems.
""",
    },

    {
        "title": "Decision Trees",
        "text": """
A decision tree predicts an output by repeatedly splitting
data according to feature values. Internal nodes represent
decision rules, branches represent outcomes, and leaf nodes
represent predictions. Splits can be selected using measures
such as information gain, entropy, or Gini impurity. Decision
trees are easy to interpret but can overfit when they become
too deep. Pruning and depth constraints can reduce overfitting.
""",
    },

    {
        "title": "Neural Networks",
        "text": """
A neural network consists of interconnected layers of
artificial neurons. A typical network contains an input layer,
one or more hidden layers, and an output layer. Each neuron
applies weights, a bias, and an activation function. During
training, backpropagation calculates gradients and an optimizer
updates the parameters. Neural networks can learn complex
nonlinear relationships and are widely used in computer vision,
natural language processing, and speech recognition.
""",
    },

    # =====================================================
    # DATA / MATHEMATICS
    # =====================================================

    {
        "title": "Linear Algebra",
        "text": """
Linear algebra studies vectors, matrices, linear transformations,
and systems of linear equations. Vectors can represent points,
features, or directions, while matrices can represent
transformations and datasets. Matrix multiplication is
fundamental to many machine learning operations. Concepts such
as eigenvalues, eigenvectors, vector spaces, and matrix
decomposition are important in machine learning and data science.
""",
    },

    {
        "title": "Probability",
        "text": """
Probability provides a mathematical framework for reasoning
about uncertainty. A probability value ranges from zero to one.
Important concepts include random variables, conditional
probability, independence, probability distributions, expected
value, and Bayes' theorem. Probability is widely used in
machine learning for modeling uncertainty and making predictions.
""",
    },

    {
        "title": "Statistics",
        "text": """
Statistics focuses on collecting, analyzing, interpreting,
and presenting data. Descriptive statistics summarize datasets
using measures such as mean, median, variance, and standard
deviation. Inferential statistics uses samples to make
conclusions about populations. Hypothesis testing and confidence
intervals are common statistical tools used in data analysis.
""",
    },

    {
        "title": "Calculus",
        "text": """
Calculus studies change and accumulation. Differentiation
measures how a function changes, while integration measures
accumulation over an interval. Derivatives are especially
important in machine learning because optimization algorithms
use gradients to adjust model parameters. Partial derivatives
and the chain rule are fundamental to training neural networks.
""",
    },

    {
        "title": "Matrix Operations",
        "text": """
Matrix operations include addition, subtraction, multiplication,
transposition, and inversion. Matrix multiplication combines
rows from one matrix with columns from another. The dimensions
of the matrices must be compatible for multiplication.
Matrix operations are central to machine learning because
datasets, weights, transformations, and embeddings are often
represented as vectors or matrices.
""",
    },

    # =====================================================
    # INFORMATION RETRIEVAL
    # =====================================================

    {
        "title": "KD-Tree",
        "text": """
A KD-tree is a space-partitioning data structure used for
organizing points in a multidimensional space. It recursively
splits the data using different dimensions at different levels.
KD-trees can accelerate nearest-neighbor and range searches
when the dimensionality is relatively low. Their effectiveness
generally decreases as dimensionality becomes very high.
""",
    },

    {
        "title": "BM25",
        "text": """
BM25 is a ranking function commonly used in information
retrieval and search engines. It scores documents based on
term frequency, inverse document frequency, and document
length normalization. Unlike simple keyword matching, BM25
reduces the influence of extremely frequent terms and accounts
for differences in document length. It is a strong traditional
baseline for text retrieval.
""",
    },

    {
        "title": "Vector Search",
        "text": """
Vector search retrieves items based on numerical representations
called embeddings. Text, images, or other objects can be mapped
into a vector space where semantically similar objects tend to
be close together. A query is converted into an embedding and
nearest-neighbor search is used to retrieve similar vectors.
Common similarity measures include cosine similarity and
Euclidean distance.
""",
    },

    {
        "title": "Hybrid Search",
        "text": """
Hybrid search combines multiple retrieval strategies to improve
search quality. A common approach combines semantic vector
retrieval with lexical retrieval such as BM25. Vector search
captures semantic similarity, while lexical search can handle
exact terminology and rare keywords effectively. Results from
different retrievers can be combined using a ranking method
such as Reciprocal Rank Fusion.
""",
    },

    {
        "title": "Reciprocal Rank Fusion",
        "text": """
Reciprocal Rank Fusion, or RRF, combines ranked result lists
from different retrieval systems. A document receives a score
based on its position in each ranking, with higher-ranked
documents receiving larger contributions. RRF is useful for
combining heterogeneous retrieval methods because the systems
do not need to produce scores on the same numerical scale.
""",
    },

    # =====================================================
    # FOOD
    # =====================================================

    {
        "title": "Pizza",
        "text": """
Pizza is a dish typically made from a flattened bread dough
base topped with ingredients such as tomato sauce, cheese,
vegetables, or meat. It is commonly baked at high temperature.
Neapolitan-style pizza traditionally uses a thin soft crust,
tomatoes, mozzarella, and basil. Different regions have developed
their own styles with different crusts, toppings, and cooking
methods.
""",
    },

    {
        "title": "Sushi",
        "text": """
Sushi is a Japanese dish centered around seasoned vinegared
rice and may include ingredients such as raw or cooked seafood,
vegetables, and seaweed. Common forms include nigiri, maki,
and temaki. Nigiri consists of hand-formed rice topped with
an ingredient, while maki is rolled with fillings and often
wrapped in nori.
""",
    },

    {
        "title": "Pasta",
        "text": """
Pasta is a staple food traditionally associated with Italian
cuisine. It can be made from wheat flour and water or eggs and
is produced in many shapes such as spaghetti, penne, fusilli,
and lasagna sheets. Pasta can be served with sauces including
tomato-based sauces, pesto, cream sauces, and meat-based sauces.
""",
    },

    {
        "title": "Indian Curry",
        "text": """
Indian curry refers to a broad family of dishes prepared with
spices, herbs, vegetables, legumes, meat, or seafood. Recipes
vary significantly between regions. Common ingredients include
cumin, coriander, turmeric, ginger, garlic, chili, and garam
masala. Curry bases can use tomatoes, onions, yogurt, coconut,
or other ingredients depending on the regional style.
""",
    },

    {
        "title": "Chocolate Cake",
        "text": """
Chocolate cake is a dessert prepared using cocoa or chocolate
along with ingredients such as flour, sugar, eggs, butter, or
oil. Different recipes produce cakes with different textures,
from light sponge cakes to dense chocolate cakes. Chocolate
cake can be finished with frosting, ganache, powdered sugar,
fruit, or other toppings.
""",
    },

    # =====================================================
    # SPORTS
    # =====================================================

    {
        "title": "Cricket",
        "text": """
Cricket is a bat-and-ball sport played between two teams.
A team attempts to score runs by striking the ball while the
opposing team tries to dismiss batters and limit scoring.
Common formats include Test cricket, One Day Internationals,
and Twenty20 cricket. The pitch contains a central strip
where the bowler delivers the ball to the batter.
""",
    },

    {
        "title": "Football",
        "text": """
Football, also called soccer in some countries, is a team
sport played between two teams of eleven players. The main
objective is to score goals by moving the ball into the
opponent's goal. Except for the goalkeeper within the penalty
area, players generally cannot deliberately use their hands.
Matches are usually divided into two halves.
""",
    },

    {
        "title": "Basketball",
        "text": """
Basketball is a team sport in which two teams attempt to score
points by shooting a ball through the opponent's hoop. Players
move the ball through passing and dribbling. Successful shots
can be worth different numbers of points depending on where
the shot is taken. Teams also compete for rebounds and defensive
possessions.
""",
    },

    {
        "title": "Tennis",
        "text": """
Tennis is a racket sport played between two opponents in
singles or two teams in doubles. Players hit a ball over a
net and attempt to land it within the opponent's court.
Matches consist of points, games, and sets. Important strokes
include the serve, forehand, backhand, volley, and overhead.
""",
    },

    {
        "title": "Olympic Games",
        "text": """
The Olympic Games are an international multi-sport competition
featuring athletes from many countries. The Summer and Winter
Games include different collections of sports. Athletes compete
for medals and represent their national teams or Olympic
delegations. The modern Olympic movement includes both
individual and team events.
""",
    },
]


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("NeuroIndex Evaluation Corpus")
    print("=" * 60)

    # Existing database
    db = DocumentDB(dim=config.DIMS)

    # Existing Ollama client
    ollama = OllamaClient()

    # -----------------------------------------------------
    # Check Ollama
    # -----------------------------------------------------

    if not ollama.is_available():

        print()
        print("ERROR: Ollama is not available.")
        print()
        print("Start Ollama and make sure your embedding")
        print("model is available.")
        print()

        return

    print()
    print("Ollama: ONLINE")
    print(
        f"Embedding model: {ollama.embed_model}"
    )

    # -----------------------------------------------------
    # Existing documents
    # -----------------------------------------------------

    existing_docs = db.list_documents()

    existing_titles = {
        doc["title"].strip().lower()
        for doc in existing_docs
    }

    print()
    print(
        f"Existing documents: {len(existing_docs)}"
    )

    # -----------------------------------------------------
    # Insert documents
    # -----------------------------------------------------

    inserted = 0
    skipped = 0
    failed = 0

    for document in DOCUMENTS:

        title = document["title"]
        text = document["text"].strip()

        # Avoid duplicate titles
        if title.strip().lower() in existing_titles:

            print(
                f"[SKIP]   {title}"
            )

            skipped += 1
            continue

        print(
            f"[EMBED]  {title}"
        )

        try:

            embedding = ollama.embed(
                text
            )

            if not embedding:

                print(
                    f"[FAILED] {title} "
                    "-> empty embedding"
                )

                failed += 1
                continue

            doc_id = str(DOCUMENTS.index(document) + 1)

            db.insert(
                doc_id,
                title,
                text,
                embedding,
            )

            print(
                f"[ADDED]  {title} "
                f"(id={doc_id})"
            )

            existing_titles.add(
                title.strip().lower()
            )

            inserted += 1

        except Exception as e:

            print(
                f"[FAILED] {title} "
                f"-> {e}"
            )

            failed += 1

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Inserted: {inserted}"
    )

    print(
        f"Skipped:  {skipped}"
    )

    print(
        f"Failed:   {failed}"
    )

    print(
        f"Total documents: {db.count()}"
    )

    print(
        f"Embedding dimensions: {db.dim}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()