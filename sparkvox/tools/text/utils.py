import re
from typing import List
from sparkvox.utils.unicode import UNICODE_ALL_CN, UNICODE_ALL_LATIN

# UNICODE_ALL_CN()

def get_word_list(text: str, simple_punctation: bool = False):
    """split text to token list

    Args:
        text (str): input text

    Returns:
        (list): token list
    """

    return  [t for t in re.findall(f'[{UNICODE_ALL_CN}]|[{UNICODE_ALL_LATIN}\'\-\d]+|[^{UNICODE_ALL_CN}{UNICODE_ALL_LATIN}\'\d]+', text)
            if re.findall(r'\S', t)]


def get_word_list(text: str) -> List[str]:
    """Split text into a list of tokens.
    
    This function splits text into tokens based on following rules:
    - Chinese characters are split into individual characters
    - Latin characters (with apostrophes, hyphens, and digits) are kept together
    - Other characters (punctuation, etc.) are separated
    
    Args:
        text (str): Input text to be tokenized
    
    Returns:
        List[str]: List of tokens
    """
    # Define the pattern components
    chinese_chars = f'[{UNICODE_ALL_CN}]'
    latin_chars = f'[{UNICODE_ALL_LATIN}\'\-\d]+'
    other_chars = f'[^{UNICODE_ALL_CN}{UNICODE_ALL_LATIN}\'\d]+'
    
    # Combine patterns
    pattern = f'{chinese_chars}|{latin_chars}|{other_chars}'
    
    # Find all matches and filter out whitespace-only tokens
    tokens = re.findall(pattern, text)
    return [token.replace(' ', '') for token in tokens if re.search(r'\S', token)]