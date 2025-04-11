
from datasets import load_dataset
import pandas as pd
import numpy as np

class TextHandler():

    def __init__(self, hf_dataset="Maximax67/English-Valid-Words", version="0.1.0"):
        self.hf_dataset = load_dataset(hf_dataset)
        
        if version == "0.1.0":
            self.load_df_english_valid_words()


    def load_df_english_valid_words(self):
        """
        Load the text database.
        """
        self.text_database = load_dataset(self.hf_dataset)
        self.df = pd.DataFrame(self.text_database['train'])
        self.df = self.df.drop(columns=['Stem', 'Stem valid probability'])
        summation = self.df["Frequency count"].sum()
        self.df["possiblity"] = self.df["Frequency count"].apply(lambda x: x/summation)

            
    def get_text(self, number_of_words=1):
        """
        Get a random text from the database.
        Args:
            number_of_words (int): The number of words to get.
        Returns:
            str: A random text from the database.
        """
        if number_of_words > 1:
            return " ".join(self.df.sample(number_of_words, weights=self.df["possiblity"])["Word"])
        else:
            return self.df.sample(1, weights=self.df["possiblity"])["Word"].values[0]
    
    def generate_text_list(self, number_of_items=10, upper_bound_words=10):
        """
        Generate a list of random texts.
        Args:
            number_of_items (int): The number of texts to generate.
            upper_bound_words (int): The upper bound of words in each text.
        Returns:
            list: A list of random texts.
        """
        texts = []
        for _ in range(number_of_items):
            number_of_words = np.random.randint(1, upper_bound_words)
            texts.append(self.get_text(number_of_words))
        return texts
    


    