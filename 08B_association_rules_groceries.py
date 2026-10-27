# Practical 8 - Version 2: Association Rules from Groceries Dataset
# Rename CSV to Groceries_dataset.csv. Install mlxtend if needed:
# pip install mlxtend

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("Groceries_dataset.csv")

# Each transaction is all items bought by one member on one date.
transactions = df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).tolist()

encoder = TransactionEncoder()
encoded = encoder.fit(transactions).transform(transactions)
basket = pd.DataFrame(encoded, columns=encoder.columns_)

frequent_items = apriori(basket, min_support=0.01, use_colnames=True)
print("Top frequent itemsets:")
print(frequent_items.sort_values("support", ascending=False).head(15))

rules = association_rules(frequent_items, metric="confidence",
                          min_threshold=0.1)
print("\nTop association rules:")
if rules.empty:
    print("No rules found. Try lowering the thresholds.")
else:
    print(rules[["antecedents", "consequents", "support",
                 "confidence", "lift"]].sort_values(
                     "lift", ascending=False).head(15))
