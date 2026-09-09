"""
Rebuilds eval_queries.json with query categories.

    keyword    - existing 90 queries, share vocabulary with the doc (easy)
    paraphrase - same question, zero shared vocabulary (stresses BM25)
    scenario   - real-world situation, doesn't name the topic (stresses both,
                 favors whichever retriever actually understands meaning)

Run once: python3 build_eval_queries.py
Then hit /doc/evaluate as usual - server.py breaks results down by category.
"""
import json

EXISTING_FILE = "eval_queries.json"

# doc_id -> (paraphrase, scenario). doc_id order matches populate_eval_data.py.
HARDER = {
    1:  ("In a structure where every node branches into at most two paths, "
         "how are smaller and larger values arranged relative to a node?",
         "I need a way to keep a sorted phonebook fast to search even as "
         "entries are added, what should I use?"),
    2:  ("Describe a sequence of connected records where each entry points "
         "to the following one.",
         "I'm building a playlist where songs are constantly inserted or "
         "removed from the middle, but I rarely jump straight to song "
         "number 50, what structure fits?"),
    3:  ("Which network exploration method visits neighbors ring by ring "
         "using a first-in first-out queue?",
         "I want the shortest number of hops between two people in a "
         "social network with no weighted connections, what technique "
         "applies?"),
    4:  ("What technique avoids redoing identical smaller calculations by "
         "caching their results?",
         "My recursive Fibonacci function is painfully slow because it "
         "repeats the same work over and over, how do I speed it up?"),
    5:  ("Which method guarantees n log n performance for arranging a list "
         "but needs extra memory?",
         "I need to alphabetize a huge list of names, extra RAM is fine, "
         "but I need a guaranteed worst-case time bound, what should I "
         "pick?"),
    6:  ("What kind of model training uses examples that already have "
         "known correct answers attached?",
         "I have a spreadsheet of houses with their sale prices and want "
         "to predict the price of a new house, what approach fits?"),
    7:  ("How does a model find structure in data when no correct answers "
         "are provided?",
         "I have a pile of customer purchase records with no labels and "
         "want to discover natural groupings among shoppers, what helps?"),
    8:  ("Which simple algorithm squashes a weighted sum of inputs into a "
         "probability for a yes-or-no outcome?",
         "I want a simple, interpretable baseline to predict whether an "
         "email is spam, what should I try first?"),
    9:  ("What model makes predictions by repeatedly asking yes-or-no "
         "questions about feature values?",
         "I need a model I can explain to a non-technical manager as a "
         "flowchart of if-then rules, what fits?"),
    10: ("What layered architecture of artificial neurons adjusts its "
         "internal weights using gradients from backpropagation?",
         "I need a model that recognizes handwritten digits from "
         "thousands of pixel images, what usually works well?"),
    11: ("Which mathematical field deals with vectors, matrices, and "
         "solving systems of linear equations?",
         "I keep seeing eigenvalues and matrix decomposition in machine "
         "learning papers, what branch of math do I need to study?"),
    12: ("What field gives tools for reasoning mathematically about "
         "uncertain or random events?",
         "I want to calculate the chance it rains given that it was "
         "cloudy this morning, what area of math handles that?"),
    13: ("Which discipline summarizes datasets and draws conclusions about "
         "a population from a sample?",
         "I surveyed 200 people, can I trust that this predicts what all "
         "10,000 employees think? What field of study helps answer that?"),
    14: ("What branch of mathematics studies rates of change and area "
         "accumulation?",
         "Training a neural network requires computing gradients to "
         "adjust weights, what math underlies that process?"),
    15: ("What operations let you combine rows and columns of numeric "
         "tables to produce a new table?",
         "I have two grids of numbers and need to combine rows from one "
         "with columns from the other, what operation do I need?"),
    16: ("What structure partitions points across multiple dimensions to "
         "speed up nearest-neighbor lookups?",
         "I have a set of 2D GPS coordinates and want to quickly find the "
         "closest point to a given location, what structure could help?"),
    17: ("Which classic scoring function ranks documents using term "
         "frequency and rarity while adjusting for length?",
         "I'm building an old-school keyword search engine and want a "
         "proven ranking formula that doesn't need embeddings, what "
         "should I use?"),
    18: ("How do systems find similar items by comparing numerical "
         "representations in a high-dimensional space?",
         "I converted my documents into embeddings and now want to find "
         "the ones closest in meaning to a new query, what technique "
         "applies?"),
    19: ("What approach blends semantic similarity retrieval with "
         "traditional keyword matching for better results?",
         "My search misses rare product codes when using embeddings "
         "alone, but keyword search misses paraphrased queries, how do I "
         "get both?"),
    20: ("What method merges several ranked lists by weighting documents "
         "based on their position in each list?",
         "I have two separate ranked result lists from different "
         "retrieval systems and want to merge them fairly, what "
         "technique fits?"),
    21: ("What baked dish starts with a flattened dough base covered in "
         "sauce, cheese, and other toppings?",
         "I want to make a traditional Italian dish with a thin soft "
         "crust, tomatoes, mozzarella and basil, what am I making?"),
    22: ("What Japanese dish is built around seasoned rice combined with "
         "seafood or vegetables and often wrapped in seaweed?",
         "I'm hand-forming rice and topping it with a slice of fish, "
         "versus rolling rice and fillings in nori, what cuisine is this?"),
    23: ("What food made from flour and water or eggs comes in many "
         "shapes and is often paired with sauce?",
         "I'm boiling long thin strands of wheat dough and topping them "
         "with a tomato-based sauce, what dish is this?"),
    24: ("What family of spiced dishes varies widely by region and often "
         "uses cumin, turmeric, and garam masala?",
         "I'm cooking a dish with a base of onions, tomatoes, and yogurt, "
         "seasoned heavily with regional spice blends, what style is "
         "this?"),
    25: ("What dessert is baked using cocoa alongside flour, sugar, eggs "
         "and butter, then finished with frosting?",
         "I want to bake a rich dessert built around cocoa and finish it "
         "with ganache, what should I make?"),
    26: ("What bat-and-ball sport has two teams, where one tries to score "
         "runs while the other tries to get batters out?",
         "I'm watching a match where a bowler delivers a ball down a "
         "central pitch strip toward someone holding a bat, what sport "
         "is this?"),
    27: ("In which team sport do eleven players per side try to put a "
         "ball into a goal mostly without using their hands?",
         "Two teams of eleven are trying to score by kicking a ball into "
         "a net, only the goalkeeper can use hands, what sport is this?"),
    28: ("What team sport involves shooting a ball through an elevated "
         "hoop to score points?",
         "Players are dribbling and passing a ball down a court trying "
         "to shoot it through a raised hoop, what sport is being played?"),
    29: ("What racket sport has players hitting a ball over a net, trying "
         "to land it inside the opponent's boundary?",
         "Two opponents are hitting a ball back and forth over a net "
         "with rackets, scoring is counted in games and sets, what sport "
         "is this?"),
    30: ("What international competition brings together athletes from "
         "many nations across a wide range of sports?",
         "Athletes from around the world are competing for medals "
         "representing their home countries across summer and winter "
         "events, what event is this?"),
}


def main():
    with open(EXISTING_FILE, "r", encoding="utf-8") as f:
        existing = json.load(f)

    queries = []
    for q in existing:
        q = dict(q)
        q.setdefault("category", "keyword")
        queries.append(q)

    for doc_id, (paraphrase, scenario) in HARDER.items():
        queries.append({
            "question": paraphrase,
            "relevant_ids": [doc_id],
            "category": "paraphrase",
        })
        queries.append({
            "question": scenario,
            "relevant_ids": [doc_id],
            "category": "scenario",
        })

    with open(EXISTING_FILE, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)

    by_cat = {}
    for q in queries:
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1

    print(f"Wrote {len(queries)} queries to {EXISTING_FILE}")
    print("By category:", by_cat)


if __name__ == "__main__":
    main()