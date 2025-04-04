import json
import colorsys
import random
from pathlib import Path

def generate_distinct_colors(n):
    """
    Generate n visually distinct colors using HSV color space
    Returns hex color codes
    """
    colors = []
    for i in range(n):
        # Use golden ratio to generate well-distributed hues
        hue = i * 0.618033988749895
        hue = hue - int(hue)
        # Use fixed saturation and value for good visibility
        saturation = 0.7
        value = 0.95
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert RGB to hex
        color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(color)
    # Shuffle colors to ensure adjacent classes have distinct colors
    random.shuffle(colors)
    return colors

def create_cvat_labels():
    """
    Create CVAT labels JSON file with unique colors for CDI words
    """
    words = [
        "chicken", "chocolate", "coffee", "coke", "cookie", "corn", "cracker", "pastry",
        "drink", "egg", "fish", "food", "grapes", "gum", "hamburger", "ice", "jello",
        "jelly", "juice", "lollipop", "meat", "melon", "milk", "muffin", "pasta", "nut",
        "orange", "alligator", "pancake", "peas", "pickle", "pizza", "popcorn", "popsicle",
        "chips", "potato", "pretzel", "animal", "pudding", "pumpkin", "raisin", "salt",
        "sandwich", "sauce", "soda", "soup", "strawberry", "ant", "toast", "tuna",
        "vanilla", "vitamin", "water", "yogurt", "beads", "belt", "bib", "boots", "bear",
        "button", "coat", "diaper", "dress", "glove", "hat", "jacket", "jeans", "mitten",
        "necklace", "bee", "pajamas", "pants", "scarf", "shirt", "shoe", "shorts", "slipper",
        "sneaker", "snowsuit", "sock", "bird", "sweater", "tights", "underpants", "zipper",
        "ankle", "arm", "butt", "cheek", "chin", "bug", "ear", "eye", "face", "finger",
        "foot", "hair", "hand", "head", "knee", "leg", "bunny", "lip", "mouth", "nose",
        "owie", "penis", "shoulder", "toe", "tongue", "tooth", "tummy", "butterfly",
        "vagina", "basket", "blanket", "bottle", "bowl", "box", "broom", "brush", "bucket",
        "camera", "cat", "can", "clock", "comb", "cup", "dish", "fork", "trash", "glass",
        "glasses", "hammer", "jar", "key", "knife", "lamp", "light", "medicine",
        "money", "mop", "nail", "napkin", "cow", "paper", "penny", "picture", "pillow",
        "plant", "plate", "purse", "radio", "scissors", "soap", "deer", "spoon", "tape",
        "telephone", "tissue", "toothbrush", "towel", "tray", "vacuum", "walker",
        "dog", "watch", "basement", "bathroom", "bathtub", "bed", "bedroom", "bench",
        "chair", "closet", "couch", "donkey", "crib", "door", "drawer", "dryer", "garage",
        "kitchen", "oven", "duck", "porch", "potty", "refrigerator", "room", "shower",
        "sink", "stairs", "stove", "elephant", "table", "tv", "window", "yard",
        "cloud", "flag", "flower", "garden", "grass", "fish", "hose", "ladder", "moon",
        "pool", "rain", "rock", "roof", "sandbox", "shovel", "frog", "sidewalk", "sky",
        "slide", "snow", "snowman", "sprinkler", "star", "stick", "stone", "street",
        "giraffe", "sun", "swing", "tree", "water", "wind", "goose", "hen", "horse",
        "kitty", "lamb", "person", "lion", "monkey", "moose", "mouse", "owl", "penguin",
        "pig", "pony", "puppy", "rooster", "sheep", "squirrel", "teddybear", "tiger",
        "turkey", "turtle", "wolf", "zebra", "airplane", "bicycle", "boat", "bus", "car",
        "firetruck", "helicopter", "motorcycle", "sled", "stroller", "tractor", "train",
        "tricycle", "truck", "ball", "balloon", "bat", "blocks", "book", "bubbles",
        "chalk", "crayon", "doll", "game", "glue", "pen", "pencil", "present", "puzzle",
        "story", "toy", "apple", "applesauce", "banana", "beans", "bread", "butter",
        "cake", "candy", "carrot", "cereal", "cheerios", "cheese"
    ]
    
    # Remove duplicates and sort
    words = sorted(set(words))
    
    # Generate colors
    colors = generate_distinct_colors(len(words))
    
    # Create labels list
    labels = []
    for word, color in zip(words, colors):
        label = {
            "name": word,
            "color": color,
            "attributes": []
        }
        labels.append(label)
    
    # Save to JSON file
    output_file = Path("cvat_cdi_labels.json")
    with output_file.open('w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"Created CVAT labels file with {len(labels)} uniquely colored labels")
    print(f"Output saved to: {output_file}")
    print(f"\nFirst few labels with their colors:")
    for label in labels[:5]:
        print(f"{label['name']}: {label['color']}")

if __name__ == "__main__":
    create_cvat_labels()