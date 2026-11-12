def load_file(doc_path):
    with open (doc_path, "r") as f:
        text=f.read()
    return text