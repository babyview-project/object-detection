"""Near-synonym groups and general words in the 295-label CDI vocabulary.

Used two ways: a verify-task foil is never a general word or in a group with the
gold label, and naming accuracy is also scored leniently (a word in the gold
label's group counts as correct).
"""

GENERAL = {"animal", "food", "toy", "drink", "dish", "person", "game", "picture",
           "room", "light", "present", "story", "money", "plant", "bug"}

GROUPS = [
    {"cat", "kitty"}, {"dog", "puppy"}, {"horse", "pony"}, {"sheep", "lamb"},
    {"bunny", "mouse"},
    {"bird", "owl", "duck", "goose", "penguin", "hen", "rooster", "chicken", "turkey"},
    {"ant", "bee", "butterfly", "bug"},
    {"rock", "stone"},
    {"shoe", "sneaker", "boots", "slipper"},
    {"coat", "jacket", "sweater", "snowsuit"},
    {"pants", "jeans", "shorts", "tights", "pajamas", "underpants"},
    {"glove", "mitten"},
    {"cup", "glass", "bottle", "jar", "can"},
    {"bowl", "dish", "plate", "tray"},
    {"soda", "coke", "drink", "juice", "milk", "water"},
    {"cookie", "cracker", "pastry", "muffin", "cake", "pretzel"},
    {"pen", "pencil", "crayon", "chalk"},
    {"towel", "napkin", "tissue", "paper", "blanket"},
    {"drawer", "closet"},
    {"bicycle", "tricycle"},
    {"truck", "firetruck", "car", "tractor", "bus"},
    {"yard", "garden", "grass", "porch"},
    {"sidewalk", "street"},
    {"teddybear", "bear", "doll", "toy"},
    {"couch", "chair", "bench"},
    {"crib", "bed"},
    {"tv", "picture"},
    {"telephone", "camera", "radio"},
    {"lamp", "light"},
    {"oven", "stove"},
    {"potty", "bathroom"},
    {"tree", "plant", "flower"},
    {"stick", "rock"},
    {"bubbles", "soap"},
]


def related(a, b):
    """True when a and b are the same word or near-synonyms."""
    return a == b or any(a in g and b in g for g in GROUPS)
