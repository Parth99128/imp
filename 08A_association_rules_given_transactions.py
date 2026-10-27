# Practical 8 - Version 1: Association Rules from Given Transactions
# Install once if needed: pip install mlxtend

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ["Bread", "Butter", "Jam", "Juice", "Milk"],
    ["Jam", "Juice", "Bread", "Butter"],
    ["Juice", "Milk", "Bread"],
    ["Butter", "Juice", "Biscuit", "Milk"],
    ["Bread", "Milk", "Butter"]
]

encoder = TransactionEncoder()
encoded = encoder.fit(transactions).transform(transactions)
basket = pd.DataFrame(encoded, columns=encoder.columns_)

frequent_items = apriori(basket, min_support=0.4, use_colnames=True)
print("Frequent itemsets:")
print(frequent_items)

rules = association_rules(frequent_items, metric="confidence",
                          min_threshold=0.6)
print("\nAssociation rules:")
if rules.empty:
    print("No rules found at this threshold.")
else:
    print(rules[["antecedents", "consequents", "support",
                 "confidence", "lift"]])
