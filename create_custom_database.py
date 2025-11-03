import sqlite3
import os

DATABASE_NAME = 'products.db'

# Your 5 products. These MUST EXACTLY MATCH the names in your labels.txt
PRODUCTS = [
    ("Apple", "Apple", 20.0),
    ("Banana", "Banana", 7.0),
    ("Peach", "Peach", 25.0),
    ("Dates", "Dates", 80.0),
    ("Cherry", "Cherry", 5.0),
]

# Delete the old database file (which had Fruits360 data)
if os.path.exists(DATABASE_NAME):
    os.remove(DATABASE_NAME)
    print(f"Removed old '{DATABASE_NAME}'.")

conn = sqlite3.connect(DATABASE_NAME)
cursor = conn.cursor()

# Create the new table
cursor.execute('''
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_name TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    price REAL NOT NULL
)
''')
print(f"Created new table 'products' in '{DATABASE_NAME}'.")

# Populate the table with your 5 items
for item in PRODUCTS:
    cursor.execute('''
    INSERT INTO products (class_name, display_name, price)
    VALUES (?, ?, ?)
    ''', (item[0], item[1], item[2]))

conn.commit()
conn.close()

print(f"Successfully populated '{DATABASE_NAME}' with {len(PRODUCTS)} custom items.")