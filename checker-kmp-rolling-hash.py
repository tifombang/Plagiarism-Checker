"""
Plagiarism Detector using Rolling Hash (Rabin-Karp) and KMP Algorithm
Author: Your Name
Date: [Date]
"""

# --- Imports ---
import tkinter as tk
from tkinter import filedialog, messagebox
import re  # Regular expressions for text cleaning
import nltk  # Natural Language Toolkit for preprocessing
from nltk.corpus import stopwords  # Stopwords list
from nltk.stem import PorterStemmer  # Stemming algorithm

# Download NLTK datasets quietly (if not already downloaded)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# --- Constants for Rolling Hash ---
# BASE: Number of possible characters (ASCII)
BASE = 256
# PRIME: A prime number to reduce hash collisions (use a larger prime like 10^9+7 for real applications)
PRIME = 101

# --- Text Preprocessing Function ---
def preprocess_text(text):
    """
    Cleans and normalizes text for comparison.
    Steps:
      1. Convert to lowercase
      2. Remove punctuation
      3. Remove stopwords (e.g., "the", "and")
      4. Stem words (e.g., "running" → "run")

    Args:
        text (str): Input text to preprocess.

    Returns:
        str: Preprocessed text.
    """
    # Step 1: Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Keep only alphanumeric and whitespace
    
    # Step 2: Remove stopwords
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Step 3: Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return " ".join(words)

# --- Rolling Hash (Rabin-Karp) Implementation ---
def compute_rolling_hashes(text, window_size):
    """
    Computes rolling hashes for all windows of size `window_size` in the text.

    Args:
        text (str): Preprocessed text.
        window_size (int): Number of characters per window.

    Returns:
        list: Rolling hash values for each window.
    """
    n = len(text)
    hashes = []
    
    if n < window_size:
        return hashes  # No windows possible
    
    # Calculate initial hash for the first window
    current_hash = 0
    for i in range(window_size):
        current_hash = (current_hash * BASE + ord(text[i])) % PRIME
    hashes.append(current_hash)
    
    # Precompute BASE^(window_size-1) mod PRIME for efficient rolling updates
    base_pow = pow(BASE, window_size - 1, PRIME)
    
    # Compute hashes for remaining windows
    for i in range(window_size, n):
        # Remove the contribution of the outgoing character (leftmost)
        outgoing_char = ord(text[i - window_size])
        # Add the contribution of the incoming character (rightmost)
        incoming_char = ord(text[i])
        
        current_hash = (current_hash - outgoing_char * base_pow) % PRIME
        current_hash = (current_hash * BASE + incoming_char) % PRIME
        
        hashes.append(current_hash)
    
    return hashes

# --- KMP Algorithm Functions ---
def compute_lps(pattern):
    """
    Computes the Longest Prefix Suffix (LPS) array for the KMP algorithm.
    LPS[i] = the longest proper prefix of pattern[0..i] which is also a suffix.

    Args:
        pattern (str): The substring to search for.

    Returns:
        list: LPS array.
    """
    lps = [0] * len(pattern)
    length = 0  # Length of the previous longest prefix suffix
    
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(text, pattern):
    """
    Searches for `pattern` in `text` using the KMP algorithm.

    Args:
        text (str): Text to search within.
        pattern (str): Substring to find.

    Returns:
        list: Indices where the pattern starts in the text.
    """
    lps = compute_lps(pattern)
    i = j = 0  # i for text, j for pattern
    matches = []
    
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
            if j == len(pattern):
                matches.append(i - j)  # Match found
                j = lps[j - 1]  # Reset using LPS
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return matches

# --- Document Comparison Logic ---
def compare_documents(doc1, doc2, window_size=20, threshold=0.8):
    """
    Compares two documents using rolling hashes and KMP verification.

    Steps:
      1. Compute rolling hashes for both documents.
      2. Build a hash map for document 1.
      3. For each hash in document 2, check collisions in document 1.
      4. Verify collisions using KMP.

    Args:
        doc1 (str): Preprocessed text of document 1.
        doc2 (str): Preprocessed text of document 2.
        window_size (int): Size of text window for hashing.
        threshold (float): Similarity threshold (0.0 to 1.0).

    Returns:
        tuple: (True/False if similar, similarity score)
    """
    # Step 1: Compute rolling hashes
    hashes1 = compute_rolling_hashes(doc1, window_size)
    hashes2 = compute_rolling_hashes(doc2, window_size)
    
    # Step 2: Build hash map for document 1
    hash_map = {}
    for idx, h in enumerate(hashes1):
        if h not in hash_map:
            hash_map[h] = []
        hash_map[h].append(idx)  # Store all positions of this hash
    
    # Step 3-4: Check hash collisions and verify with KMP
    total_matches = 0
    for idx2, h in enumerate(hashes2):
        if h in hash_map:
            # Check all positions in doc1 with this hash
            for idx1 in hash_map[h]:
                # Extract the windows from both documents
                window1 = doc1[idx1 : idx1 + window_size]
                window2 = doc2[idx2 : idx2 + window_size]
                # Verify with KMP to avoid false positives
                if kmp_search(window2, window1):
                    total_matches += 1
                    break  # Avoid counting multiple matches for the same window
    
    # Calculate similarity score
    similarity = total_matches / len(hashes1) if hashes1 else 0
    return similarity >= threshold, similarity

# ----------------- GUI Implementation -----------------
def open_file_dialog(file_path_var):
    """
    Opens a file dialog and updates the provided StringVar with the selected file path.

    Args:
        file_path_var (tk.StringVar): Variable to store the selected file path.
    """
    filepath = filedialog.askopenfilename(
        title="Select a Text File", filetypes=(("Text Files", "*.txt"),))
    if filepath:
        file_path_var.set(filepath)

def compare_documents_gui():
    """Handles document comparison when the GUI 'Compare' button is clicked."""
    file_path1 = file_path1_var.get()
    file_path2 = file_path2_var.get()

    # Validate file selection
    if not file_path1 or not file_path2:
        messagebox.showerror("Error", "Please select both document files.")
        return

    # Read files
    try:
        with open(file_path1, "r", encoding="utf-8") as file1:
            doc1_content = file1.read()
        with open(file_path2, "r", encoding="utf-8") as file2:
            doc2_content = file2.read()
    except Exception as e:
        messagebox.showerror("Error", f"Error opening file: {e}")
        return

    # Preprocess documents
    doc1_clean = preprocess_text(doc1_content)
    doc2_clean = preprocess_text(doc2_content)

    # Get parameters
    try:
        window_size = int(window_size_entry.get())
        threshold = float(threshold_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Invalid window size or threshold.")
        return

    # Compare documents
    are_similar, similarity_score = compare_documents(doc1_clean, doc2_clean, window_size, threshold)

    # Display results
    result_label.config(
        text=f"Similarity: {similarity_score:.2f}. Documents are {'similar' if are_similar else 'not similar'}."
    )

# --- GUI Setup ---
root = tk.Tk()
root.title("Plagiarism Detector")

# Document 1 Selection
file_path1_var = tk.StringVar()
file1_label = tk.Label(root, text="Document 1:")
file1_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
file1_entry = tk.Entry(root, textvariable=file_path1_var, width=50)
file1_entry.grid(row=0, column=1, padx=5, pady=5)
file1_button = tk.Button(root, text="Browse", command=lambda: open_file_dialog(file_path1_var))
file1_button.grid(row=0, column=2, padx=5, pady=5)

# Document 2 Selection
file_path2_var = tk.StringVar()
file2_label = tk.Label(root, text="Document 2:")
file2_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)
file2_entry = tk.Entry(root, textvariable=file_path2_var, width=50)
file2_entry.grid(row=1, column=1, padx=5, pady=5)
file2_button = tk.Button(root, text="Browse", command=lambda: open_file_dialog(file_path2_var))
file2_button.grid(row=1, column=2, padx=5, pady=5)

# Parameters: Window Size
window_size_label = tk.Label(root, text="Window Size (characters):")
window_size_label.grid(row=2, column=0, sticky="w", padx=5, pady=5)
window_size_entry = tk.Entry(root, width=10)
window_size_entry.insert(0, "20")  # Default value
window_size_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

# Parameters: Threshold
threshold_label = tk.Label(root, text="Similarity Threshold (0-1):")
threshold_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
threshold_entry = tk.Entry(root, width=10)
threshold_entry.insert(0, "0.8")  # Default value
threshold_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

# Compare Button
compare_button = tk.Button(root, text="Compare Documents", command=compare_documents_gui)
compare_button.grid(row=4, column=0, columnspan=3, pady=10)

# Result Display
result_label = tk.Label(root, text="", fg="blue")
result_label.grid(row=5, column=0, columnspan=3)

# Start the GUI
root.mainloop()