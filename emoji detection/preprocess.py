import re
import html

# Dictionary of emoji metadata: label -> (emoji, name)
EMOJI_METADATA = {
    0: ("😜", "Winking Face with Tongue"),
    1: ("📸", "Camera with Flash"),
    2: ("😍", "Heart Eyes"),
    3: ("😂", "Face with Tears of Joy"),
    4: ("😉", "Winking Face"),
    5: ("🎄", "Christmas Tree"),
    6: ("📷", "Camera"),
    7: ("🔥", "Fire"),
    8: ("😘", "Face Blowing a Kiss"),
    9: ("❤️", "Red Heart"),
    10: ("😁", "Beaming Face with Smiling Eyes"),
    11: ("🇺🇸", "United States Flag"),
    12: ("☀️", "Sun / Sunshine"),
    13: ("✨", "Sparkles"),
    14: ("💙", "Blue Heart"),
    15: ("💕", "Two Hearts"),
    16: ("😎", "Smiling Face with Sunglasses"),
    17: ("😊", "Smiling Face with Smiling Eyes"),
    18: ("💜", "Purple Heart"),
    19: ("💯", "Hundred Points"),
    20: ("😭", "Loudly Crying Face"),
    21: ("🥳", "Partying Face"),
    22: ("😡", "Enraged / Angry Face"),
    23: ("🙏", "Folded Hands / Prayer / Thanks"),
    24: ("💪", "Flexed Biceps / Gym / Fitness"),
    25: ("👏", "Clapping Hands / Applause"),
    26: ("😴", "Sleeping Face / Tired"),
    27: ("🎂", "Birthday Cake / Celebration"),
    28: ("☕", "Hot Beverage / Coffee"),
    29: ("🚀", "Rocket / Launch / Moon")
}

# Emoticon regex pattern mapping to capture direct emoji equivalents in text
EMOTICON_MAP = {
    r'(?::\)|:-\)|=\))': ' emoticon_smile ',
    r'(?:;\)|;-\))': ' emoticon_wink ',
    r'(?::D|:-D|=D)': ' emoticon_laugh ',
    r'(?::P|:-P|;P|;-P)': ' emoticon_tongue ',
    r'(?:<3)': ' emoticon_heart ',
    r'(?:B-\)|8-\))': ' emoticon_cool ',
    r'(?::\*|:-\*)': ' emoticon_kiss ',
    r'(?:;_;|T_T|:\'\(|:\'-\()': ' emoticon_crying ',
    r'(?:>:\(|>:-\()': ' emoticon_angry '
}

def segment_hashtag(tag: str) -> str:
    """Splits camelCase or alphanumeric hashtags: #MerryChristmas -> Merry Christmas"""
    t = tag.lstrip('#')
    t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
    t = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', t)
    t = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', t)
    return t

def clean_text(text: str) -> str:
    """
    Advanced text cleaning & feature enhancement for emoji prediction:
    - Decodes HTML entities (e.g., &amp; -> &)
    - Maps direct textual emoticons to tokens
    - Segments camelCase hashtags into constituent words while preserving original tags
    - Normalizes elongated character sequences (e.g., 'sooooo' -> 'sooo')
    - Strips @mentions and URLs
    - Normalizes whitespace and converts to lowercase
    """
    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Map text emoticons
    for pattern, token in EMOTICON_MAP.items():
        text = re.sub(pattern, token, text)

    # Segment and expand hashtags
    def expand_ht(m):
        raw = m.group(0)
        seg = segment_hashtag(raw)
        return f"{raw} {seg}"

    text = re.sub(r'#[A-Za-z0-9_]+', expand_ht, text)

    # Remove @mentions
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)

    # Remove web links
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Normalize character repetitions (e.g., 'yaaaay' -> 'yaay')
    text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)

    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text.lower()
