
from datasets import load_dataset


class TextHandler():

    def __init__(self, hf_dataset="Maximax67/English-Valid-Words", version="0.1.0"):
        self.hf_dataset = load_dataset(hf_dataset)
        
        if version == "0.1.0":
            self.load_text_database()

        def load_database(self):
            """
            Load the text database.
            """
            self.text_database = load_dataset(self.hf_dataset)
            self.text_database = self.text_database["train"]["text"]
            self.text_database = list(set(self.text_database))
            print("Text database loaded successfully.")

    def gen_text(self, length=10):


    