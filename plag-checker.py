# An AI-based Plagiarism checker application 


# -- Importing the necessary libraries
import hashlib
import tkinter as tk
from tkinter import filedialog, messagebox
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources (do this once, if needed)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    PorterStemmer()
except LookupError:
    nltk.download('punkt')


# Preprocessing of the Documents
# - lowercase, tokenization, stopword removal, etc. - #

def preprocess_text(text):
    """Preprocesses the text for plagiarism detection."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)


def rolling_hash_sha256(text, window_size):
    """Calculates rolling hashes using SHA256."""
    hashes = []
    for i in range(len(text) - window_size + 1):
        window_text = text[i:i + window_size].encode('utf-8')
        current_hash = hashlib.sha256(window_text).hexdigest()
        hashes.append(current_hash)
    return hashes


def compare_documents_rolling_hash(doc1, doc2, window_size=20, threshold=0.8):
    """Compares two documents using rolling hashes."""
    hashes1 = rolling_hash_sha256(doc1, window_size)
    hashes2 = rolling_hash_sha256(doc2, window_size)

    matches = 0
    min_len = min(len(hashes1), len(hashes2))

    for i in range(min_len):
        if hashes1[i] == hashes2[i]:
            matches += 1

    similarity = 0
    if min_len > 0:
        similarity = matches / min_len

    return similarity >= threshold, similarity


def open_file_dialog(file_path_var):
    """Opens the file dialog and updates the entry (only when button is clicked)."""
    filepath = filedialog.askopenfilename(
        title="Select a Text File", filetypes=(("Text Files", "*.txt"),)
    )
    if filepath:
        file_path_var.set(filepath)


def compare_documents():
    """Compares the documents."""
    file_path1 = file_path1_var.get()
    file_path2 = file_path2_var.get()

    if not file_path1 or not file_path2:
        messagebox.showerror("Error", "Please select both document files.")
        return

    try:
        with open(file_path1, "r", encoding="utf-8") as file1:
            doc1_content = file1.read()
        with open(file_path2, "r", encoding="utf-8") as file2:
            doc2_content = file2.read()
    except FileNotFoundError:
        messagebox.showerror("Error", "One or both files not found.")
        return
    except Exception as e:
        messagebox.showerror("Error", f"Error opening file: {e}")
        return

    doc1_content = preprocess_text(doc1_content)
    doc2_content = preprocess_text(doc2_content)

    try:
        window_size = int(window_size_entry.get())
        threshold = float(threshold_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Invalid window size or threshold.")
        return

    are_similar, similarity_score = compare_documents_rolling_hash(doc1_content, doc2_content, window_size, threshold)

    result_label.config(
        text=f"Similarity: {similarity_score:.2f}.  {'Oh snap!!! Documents are similar' if are_similar else 'Documents are not similar'}."
    )


# --- GUI setup using Tkinter ---

root = tk.Tk()
root.title("Plagiarism Checker")

# File selection (Document 1)
file_path1_var = tk.StringVar()
file1_label = tk.Label(root, text="Document 1:")
file1_label.grid(row=0, column=0, sticky="w")
file1_entry = tk.Entry(root, textvariable=file_path1_var, width=50)
file1_entry.grid(row=0, column=1, padx=5, pady=5)
file1_button = tk.Button(root, text="Browse", command=lambda: open_file_dialog(file_path1_var))
file1_button.grid(row=0, column=2, padx=5, pady=5)

# File selection (Document 2)
file_path2_var = tk.StringVar()
file2_label = tk.Label(root, text="Document 2:")
file2_label.grid(row=1, column=0, sticky="w")
file2_entry = tk.Entry(root, textvariable=file_path2_var, width=50)
file2_entry.grid(row=1, column=1, padx=5, pady=5)
file2_button = tk.Button(root, text="Browse", command=lambda: open_file_dialog(file_path2_var))
file2_button.grid(row=1, column=2, padx=5, pady=5)

# Parameters
window_size_label = tk.Label(root, text="Text block Size:")
window_size_label.grid(row=2, column=0, sticky="w")
window_size_entry = tk.Entry(root, width=10)
window_size_entry.insert(0, "20")  # Default value
window_size_entry.grid(row=2, column=1, padx=5, pady=5)

threshold_label = tk.Label(root, text="Sensitivity (0-1):")
threshold_label.grid(row=3, column=0, sticky="w")
threshold_entry = tk.Entry(root, width=10)
threshold_entry.insert(0, "0.8")  # Default value
threshold_entry.grid(row=3, column=1, padx=5, pady=5)

# Compare button
compare_button = tk.Button(root, text="Compare", command=compare_documents)
compare_button.grid(row=4, column=0, columnspan=3, pady=10)

# Result label
result_label = tk.Label(root, text="")
result_label.grid(row=5, column=0, columnspan=3)

root.mainloop()