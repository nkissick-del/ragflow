import sys
import os
import traceback


# Add current directory to path to import rag modules
sys.path.append(os.getcwd())

from rag.templates.tag import chunk


def dummy_callback(prog, msg):
    print(f"PROGRESS: {prog:.2f} - {msg}")


# Create a dummy CSV file
csv_content = """content1,tag1
content2,tag2
multiline
content,tag3
"""
with open("test_tag.csv", "w") as f:
    f.write(csv_content)

print("--- Testing CSV Parsing ---")
try:
    res = chunk("test_tag.csv", callback=dummy_callback)
    for doc in res:
        print(f"Doc: {doc['content_with_weight']} | Tags: {doc['tag_kwd']}")
except Exception as e:
    print(f"Error during CSV chunk: {e}")
    traceback.print_exc()

# Create a dummy TXT file
txt_content = """content_a\ttag_a
content_b\ttag_b
extra_line
content_c\ttag_c
"""
with open("test_tag.txt", "w") as f:
    f.write(txt_content)

print("\n--- Testing TXT Parsing ---")
try:
    res = chunk("test_tag.txt", callback=dummy_callback)
    for doc in res:
        print(f"Doc: {doc['content_with_weight']} | Tags: {doc['tag_kwd']}")
except Exception as e:
    print(f"Error during TXT chunk: {e}")
    traceback.print_exc()

# Cleanup
try:
    if os.path.exists("test_tag.csv"):
        os.remove("test_tag.csv")
    if os.path.exists("test_tag.txt"):
        os.remove("test_tag.txt")
except Exception as e:
    print(f"Error during cleanup: {e}")
    traceback.print_exc()
