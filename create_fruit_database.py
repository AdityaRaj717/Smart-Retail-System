import sqlite3
import os
import re

# --- CONFIGURATION ---

# 1. IMPORTANT: Set this variable to the exact path of your
#    Fruits360 'Training' or 'Test' directory.
#    It must be the folder that CONTAINS all the fruit subfolders.
#
#    Examples:
#    FRUITS360_TRAINING_PATH = 'C:/Users/YourUser/Desktop/fruits-360/Training'
#    FRUITS360_TRAINING_PATH = '/home/youruser/datasets/fruits-360/Training'
#
FRUITS360_TRAINING_PATH = '/mnt/Personal/Programming/Self Learning/Projects/Capstone/Smart Retail Checkout System/dataset/fruits-360_100x100/fruits-360/Training'  # <-- !!! UPDATE THIS LINE !!!

# 2. This will be the name of your new database file.
DATABASE_NAME = 'products.db'

# 3. Set a default price for all fruits.
#    You will need to update this manually in the database file later.
DEFAULT_PRICE = 10.0  # e.g., 10.0 currency units

# --- END CONFIGURATION ---


def get_fruit_names(dataset_path):
    """Scans the dataset path and returns a list of fruit directory names."""
    if not os.path.exists(dataset_path):
        print(f"--- ERROR ---")
        print(f"Dataset path not found at: '{dataset_path}'")
        print("Please update the 'FRUITS360_TRAINING_PATH' variable in this script.")
        return None

    print(f"Scanning for fruit folders in: {dataset_path}")
    fruit_names = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            fruit_names.append(item)

    if not fruit_names:
        print(f"--- ERROR ---")
        print(f"No subdirectories (fruit folders) found in '{dataset_path}'.")
        print("Make sure this path points directly to the 'Training' folder.")
        return None

    return fruit_names

def generate_display_name(class_name):
    """Creates a prettier 'display_name' from a folder 'class_name'."""
    # Replaces underscores with spaces
    display_name = class_name.replace('_', ' ')

    # Optional: Adds a space between text and numbers (e.g., "AppleGolden 1" -> "AppleGolden 1")
    display_name = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', display_name)

    # Optional: Adds a space between lowercase and uppercase (e.g., "AppleRed" -> "Apple Red")
    display_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', display_name)

    return display_name.strip()

def create_db(fruit_list):
    """Creates and populates the SQLite database."""

    # Delete the old database file if it exists, since we have a new schema
    if os.path.exists(DATABASE_NAME):
        print(f"Found existing '{DATABASE_NAME}'. Deleting it to create a new one.")
        os.remove(DATABASE_NAME)

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    # Create the 'products' table
    # - class_name: The exact folder name (e.g., "Apple_Golden_1"). This is our lookup key.
    # - display_name: A prettier name for the bill (e.g., "Apple Golden 1").
    # - price: The price of the item.
    try:
        cursor.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            price REAL NOT NULL
        )
        ''')
        print(f"Created new table 'products' in '{DATABASE_NAME}'.")
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")
        conn.close()
        return

    # Populate the table
    print(f"Populating table with {len(fruit_list)} items...")
    for class_name in fruit_list:
        display_name = generate_display_name(class_name)

        try:
            cursor.execute('''
            INSERT INTO products (class_name, display_name, price)
            VALUES (?, ?, ?)
            ''', (class_name, display_name, DEFAULT_PRICE))
        except sqlite3.Error as e:
            print(f"Error inserting {class_name}: {e}")

    conn.commit()
    conn.close()

    print(f"\n--- SUCCESS ---")
    print(f"Successfully created '{DATABASE_NAME}' with {len(fruit_list)} fruit items.")
    print(f"All items have been set to a default price of {DEFAULT_PRICE}.")

def main():
    fruit_names = get_fruit_names(FRUITS360_TRAINING_PATH)
    if fruit_names:
        print(f"Found {len(fruit_names)} fruit types in the dataset.")
        create_db(fruit_names)

if __name__ == "__main__":
    main()