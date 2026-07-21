import math
from collections import Counter

def entropy(labels):

    total = len(labels)

    counts = Counter(labels)

    ent = 0

    for count in counts.values():

        p = count / total

        ent -= p * math.log2(p)

    return ent

def information_gain(X, y, feature_index):

    # Entropy of complete dataset
    total_entropy = entropy(y)

    # Find unique values of the selected feature
    values = set()

    for row in X:
        values.add(row[feature_index])

    weighted_entropy = 0

    # Calculate entropy for each split
    for value in values:

        sub_y = []

        # Collect class labels belonging to this value
        for i in range(len(X)):

            if X[i][feature_index] == value:

                sub_y.append(y[i])

        weight = len(sub_y) / len(y)

        weighted_entropy = weighted_entropy + weight * entropy(sub_y)

    information_gain = total_entropy - weighted_entropy

    return information_gain

# ID3 Algorithm
def id3(X, y, features):

    # Step 1: If all labels are same
    if len(set(y)) == 1:
        return y[0]

    # Step 2: No features left
    if not features:
        return Counter(y).most_common(1)[0][0]

    # Step 3 & 4: Calculate Information Gain
    gains = {}

    for feature in features:
        gains[feature] = information_gain(X, y, feature)

    # Step 5: Best Feature
    best_feature = max(gains, key=gains.get)

    # Step 6: Create Node
    tree = {best_feature: {}}

    # Step 7: Split Dataset
    values = set()

    for row in X:
        values.add(row[best_feature])

    for value in values:

        X_sub = []
        y_sub = []

        for i in range(len(X)):
            if X[i][best_feature] == value:
                X_sub.append(X[i])
                y_sub.append(y[i])

        remaining_features = features.copy()
        remaining_features.remove(best_feature)

        tree[best_feature][value] = id3(
            X_sub,
            y_sub,
            remaining_features
        )

    return tree


def predict(tree, sample):

    # Keep checking until we reach a leaf
    while isinstance(tree, dict):

        # Get the feature at the current node
        for feature in tree:
            break

        # Get the sample's value for that feature
        value = sample[feature]

        # Move to the next branch if it exists
        if value in tree[feature]:
            tree = tree[feature][value]
        else:
            return "Unknown"

    # When tree is no longer a dictionary, it is the prediction
    return tree


# ------------------------------
# Dataset
# ------------------------------

X_train = [
    ['Y', 'High', 'no', 'fair'],
    ['Y', 'High', 'no', 'excellent'],
    ['M_A', 'High', 'no', 'fair'],
    ['S', 'medium', 'no', 'fair'],
    ['S', 'low', 'yes', 'fair'],
    ['S', 'low', 'yes', 'excellent'],
    ['M_A', 'low', 'yes', 'excellent'],
    ['Y', 'medium', 'no', 'fair'],
    ['Y', 'low', 'yes', 'fair'],
    ['S', 'medium', 'yes', 'fair'],
    ['Y', 'medium', 'yes', 'excellent'],
    ['M_A', 'medium', 'no', 'excellent'],
    ['M_A', 'High', 'yes', 'fair'],
    ['S', 'medium', 'no', 'excellent']
]

y_train = [
    'no', 'no', 'yes', 'yes',
    'yes', 'no', 'yes', 'no',
    'yes', 'yes', 'yes', 'yes',
    'yes', 'no'
]

# Feature indices
features = [0, 1, 2, 3]

# Build Tree
tree = id3(X_train, y_train, features)

print("Decision Tree:")
print(tree)

# Test Sample
test_sample = ['Y', 'low', 'yes', 'fair']

prediction = predict(tree, test_sample)

print("\nPrediction:", prediction)
