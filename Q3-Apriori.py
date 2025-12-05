"mam gave this code " 
# -------------------------------------------
# APRIORI FOR THE GIVEN DATASET (T1 to T5)
# -------------------------------------------

# Install library (Colab only)
# !pip install mlxtend pandas
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jupyter_client.session")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Given dataset
dataset = [
    ['milk', 'bread', 'nuts', 'apple'],   # T1
    ['milk', 'bread', 'apple'],           # T2
    ['milk', 'bread'],                    # T3
    ['milk', 'bread', 'apple'],           # T4
    ['milk', 'bread', 'nuts']             # T5
]

# Convert to one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("Transaction Dataset:")
print(df)

# -------------------------------------------
# CASE (a): min support = 50%, confidence = 75%
# -------------------------------------------
print("\n=== CASE (a): Support = 0.50, Confidence = 0.75 ===")

freq_a = apriori(df, min_support=0.50, use_colnames=True)
print("\nFrequent Itemsets (Case A):")
print(freq_a)

rules_a = association_rules(freq_a, metric="confidence", min_threshold=0.75)
print("\nAssociation Rules (Case A):")
print(rules_a[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# -------------------------------------------
# CASE (b): min support = 60%, confidence = 60%
# -------------------------------------------
print("\n=== CASE (b): Support = 0.60, Confidence = 0.60 ===")

freq_b = apriori(df, min_support=0.60, use_colnames=True)
print("\nFrequent Itemsets (Case B):")
print(freq_b)

rules_b = association_rules(freq_b, metric="confidence", min_threshold=0.60)
print("\nAssociation Rules (Case B):")
print(rules_b[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


