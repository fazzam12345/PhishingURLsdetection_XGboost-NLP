import pandas as pd
from urllib.parse import urlparse
import tldextract
import logging
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

class FeatureEngineering:
    """
    Class to handle feature engineering for URL data.
    """

    def __init__(self, cv=None):
        """Initializes the FeatureEngineering class with pre-trained tokenizer and CountVectorizer."""
        self.stemmer = SnowballStemmer("english")
        self.tokenizer = RegexpTokenizer(r"[A-Za-z]+")
        if cv is None:
            self.cv = CountVectorizer(max_features=25, ngram_range=(1, 3))
        else:
            self.cv = cv

    def _preprocess_text(self, url: str) -> str:
        """
        Tokenizes and applies stemming to the text of a URL.

        Args:
            url (str): The URL to preprocess.

        Returns:
            str: The preprocessed text.
        """
        tokens = self.tokenizer.tokenize(url)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return " ".join(stemmed_tokens)

    def extract_custom_features(self, url: str) -> dict:
        """
        Extracts custom features from a URL.

        Args:
            url (str): The URL to extract features from.

        Returns:
            dict: A dictionary of extracted features.
        """
        features = {}
        
        try:
            parsed_url = urlparse(url)
            extracted = tldextract.extract(url)
        except Exception as e:
            logging.error(f"Error parsing URL {url}: {e}")
            return {}
        
        # URL Length
        features["url_length"] = len(url)
        features["hostname_length"] = len(parsed_url.netloc)
        features["path_length"] = len(parsed_url.path)

        # Domain
        features["num_subdomains"] = len(extracted.subdomain.split(".")) if extracted.subdomain else 0
        features["domain_length"] = len(extracted.domain)
        features["suffix_length"] = len(extracted.suffix)

        # Character Density
        features["digit_density"] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
        features["letter_density"] = sum(c.isalpha() for c in url) / len(url) if len(url) > 0 else 0
        features["special_char_density"] = sum(not c.isalnum() for c in url) / len(url) if len(url) > 0 else 0
        features["uppercase_density"] = sum(c.isupper() for c in url) / len(url) if len(url) > 0 else 0
        features["lowercase_density"] = sum(c.islower() for c in url) / len(url) if len(url) > 0 else 0

        # Protocol and Subdomain
        features["has_https"] = int(parsed_url.scheme == "https")
        features["has_www"] = int("www" in url.lower())

        # Suspicious Words
        features["num_slashes"] = url.count("/")
        features["query_length"] = len(parsed_url.query)
        features["has_query"] = int(bool(parsed_url.query))

        # Vowel and Consonant Density
        features["vowel_density"] = sum(1 for c in url if c in "aeiouAEIOU") / len(url) if len(url) > 0 else 0
        features["consonant_density"] = sum(1 for c in url if c.isalpha() and c.lower() not in "aeiou") / len(url) if len(url) > 0 else 0

        # Entropy
        char_counts = [url.count(c) for c in set(url)]
        features["entropy"] = -sum((np.array(char_counts) / len(url)) * np.log(np.array(char_counts) / len(url)))

        # Phishing Words
        phishing_words = [
            "paypal", "login", "signin", "verify", "account", "update", "payment", 
            "bank", "secure", "alert", "confirm", "password", "financial", "webscr", 
            "submit"
        ]
        features["has_phishing_words"] = int(any(word in url.lower() for word in phishing_words))
        features["phishing_word_count"] = sum(url.lower().count(word) for word in phishing_words)
        features["phishing_word_density"] = features["phishing_word_count"] / len(url) if len(url) > 0 else 0

        # Additional Features
        features["num_dashes"] = url.count("-")
        features["num_dots"] = url.count(".")
        features["num_at_symbols"] = url.count("@")
        features["num_and_symbols"] = url.count("&")
        features["num_equals_symbols"] = url.count("=")
        features["num_underscore_symbols"] = url.count("_")

        # Check for IP address in the domain
        features["is_ip"] = int(any(char.isdigit() for char in extracted.domain.split('.')))

        # Additional Out-of-the-Box Features
        features["has_port"] = int(parsed_url.port is not None)
        features["has_fragment"] = int(bool(parsed_url.fragment))
        features["has_params"] = int(bool(parsed_url.params))
        features["has_username"] = int(bool(parsed_url.username))
        features["has_password"] = int(bool(parsed_url.password))
        features["has_hexadecimal"] = int(any(c in "0123456789abcdefABCDEF" for c in url))
        features["has_base64"] = int(any(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in url))
        features["has_unicode"] = int(any(ord(c) > 127 for c in url))
        

        return features

    def extract_tokenizer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts tokenizer-based features from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing URLs.

        Returns:
            pd.DataFrame: The DataFrame with tokenizer-based features.
        """
        df["text_processed"] = df["domain"].apply(self._preprocess_text)
        self.cv.fit(df["text_processed"])
        pickle.dump(self.cv, open("../mlops/artifacts/cv.pkl", "wb"))

        tokenizer_features = []
        for _, row in df.iterrows():
            features = {}
            text_features = self.cv.transform([row["text_processed"]])
            text_features_dict = {
                f"nlp_{i}": val
                for i, val in enumerate(text_features.toarray()[0])
            }
            features.update(text_features_dict)
            tokenizer_features.append(features)

        return pd.DataFrame(tokenizer_features)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts all features from a DataFrame containing URLs.

        Args:
            df (pd.DataFrame): The input DataFrame with a 'domain' column.

        Returns:
            pd.DataFrame: DataFrame with all extracted features.
        """

        custom_features = df["domain"].apply(self.extract_custom_features).apply(pd.Series)
        tokenizer_features = self.extract_tokenizer_features(df)

        # Combine features
        all_features = pd.concat([custom_features, tokenizer_features], axis=1)
        all_features["label"] = df["label"]
        return all_features
        
    def save_features(self, df: pd.DataFrame, file_path: str):
        """
        Saves the extracted features to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame containing the extracted features.
            file_path (str): The path to save the CSV file.
        """
        try:
            df.to_csv(file_path, index=False)
            print(f"Features saved to CSV file: {file_path}")
        except PermissionError:
            print(f"Permission denied when trying to save to {file_path}. Please check your permissions.")
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")
            
    def feature_engineering_streamlit(self, url_input) -> pd.DataFrame:
        """Extracts features for URLs.

        Args:
            url_input (str or pd.DataFrame): 
                - If string, it's assumed to be a single URL.
                - If DataFrame, it's assumed to have a 'domain' column.

        Returns:
            pd.DataFrame: DataFrame with extracted features for each URL.
        """
        if isinstance(url_input, str):
            # Single URL case
            custom_features = self.extract_custom_features(url_input)
            processed_text = self._preprocess_text(url_input)
            text_features = self.cv.transform([processed_text])
            text_features_dict = {
                f"nlp_{i}": val for i, val in enumerate(text_features.toarray()[0])
            }
            custom_features.update(text_features_dict)
            return pd.DataFrame([custom_features]) 
        elif isinstance(url_input, pd.Series):
            # Handle Series of URLs
            all_features = []
            for url in url_input:
                features = self.feature_engineering_streamlit(url)  # Call itself for each URL
                all_features.append(features)
            return pd.concat(all_features, ignore_index=True)
        elif isinstance(url_input, pd.DataFrame):
            # DataFrame case
            all_features = []
            for _, row in url_input.iterrows():
                url = row['domain'] 
                features = self.feature_engineering_streamlit(url)
                all_features.append(features)
            return pd.concat(all_features, ignore_index=True)
        else:
            raise TypeError("Input must be a string (single URL) or a pandas DataFrame with a 'domain' column.")