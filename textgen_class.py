

class TextGen():

    def __init__(self, pexel_api_key: str, version: str = "0.1.0"):
        """
        Initialize the TextGen class with a model name and optional tokenizer name.
        """
        self.pexel_api_key = pexel_api_key
        # Load the text database
        if version == "0.1.0":
            self.load_text_database(text_database_hf="Maximax67/English-Valid-Words")

    def download_pics(self, queries: list = ["blank paper"]):
        pass

    def load_model(self, model_name: str):
        """
        Load the specified model and tokenizer.
        """
        pass

    def load_text_database(self, text_database_hf: str = "Maximax67/English-Valid-Words"):
        """
        Load the text database.
        """
        from datasets import load_dataset
        self.text_database = load_dataset(text_database_hf)
        

