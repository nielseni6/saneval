import re
from typing import Dict, List, Optional, Union

# Utility constants
COMMON_QUANTITIES = {
    "a": 1,
    "an": 1,
    "one": 1,
    "single": 1,
    "two": 2,
    "couple": 2,
    "pair": 2,
    "both": 2,
    "three": 3,
    "few": 3,
    "several": 4,
    "many": 10,
    "dozen": 12,
    "dozens": 12,
    "hundred": 100,
    "hundreds": 100,
}

ORDINAL_WORDS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth",
}


def number_to_word(num: Union[int, float, str]) -> str:
    """
    Convert integer, float, or string number to word form.

    Args:
        num: Number to convert (int, float, or string)

    Returns:
        String representation of the number in words

    Examples:
        >>> number_to_word(42)
        'forty-two'
        >>> number_to_word(1.5)
        'one and a half'
        >>> number_to_word("123")
        'one hundred twenty-three'
    """
    try:
        # Handle string input
        if isinstance(num, str):
            num = num.strip()
            if "." in num:
                num = float(num)
            else:
                num = int(num)

        # Handle float input
        if isinstance(num, float):
            return float_to_word(num)

        # Handle negative numbers
        if num < 0:
            return f"negative {number_to_word(abs(num))}"

        # Handle zero
        if num == 0:
            return "zero"

        # Handle large numbers
        if num >= 1000000000000:  # trillion
            return large_number_to_word(num)

        return int_to_word(num)

    except (ValueError, TypeError):
        return str(num)  # Fallback to string representation


def int_to_word(num: int) -> str:
    """Convert integer to word form with comprehensive coverage."""
    if num == 0:
        return "zero"

    # Basic numbers 0-20
    ones = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]

    # Tens
    tens = [
        "",
        "",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]

    if num < 20:
        return ones[num]
    elif num < 100:
        return tens[num // 10] + ("" if num % 10 == 0 else "-" + ones[num % 10])
    elif num < 1000:
        return (
            ones[num // 100]
            + " hundred"
            + ("" if num % 100 == 0 else " " + int_to_word(num % 100))
        )
    elif num < 1000000:
        return (
            int_to_word(num // 1000)
            + " thousand"
            + ("" if num % 1000 == 0 else " " + int_to_word(num % 1000))
        )
    elif num < 1000000000:
        return (
            int_to_word(num // 1000000)
            + " million"
            + ("" if num % 1000000 == 0 else " " + int_to_word(num % 1000000))
        )
    elif num < 1000000000000:
        return (
            int_to_word(num // 1000000000)
            + " billion"
            + ("" if num % 1000000000 == 0 else " " + int_to_word(num % 1000000000))
        )
    else:
        return large_number_to_word(num)


def float_to_word(num: float) -> str:
    """Convert float to word form."""
    if num == int(num):
        return int_to_word(int(num))

    # Handle common fractions
    decimal_part = num - int(num)

    # Check for common fractions
    fraction_words = {
        0.5: "and a half",
        0.25: "and a quarter",
        0.75: "and three quarters",
        0.33: "and a third",
        0.67: "and two thirds",
        0.2: "and a fifth",
        0.4: "and two fifths",
        0.6: "and three fifths",
        0.8: "and four fifths",
    }

    # Check if decimal part matches common fractions (with tolerance)
    for frac, word in fraction_words.items():
        if abs(decimal_part - frac) < 0.01:
            return f"{int_to_word(int(num))} {word}"

    # Handle general decimal cases
    decimal_str = f"{decimal_part:.2f}"[2:]  # Get decimal digits
    return f"{int_to_word(int(num))} point {' '.join(int_to_word(int(d)) for d in decimal_str)}"


def large_number_to_word(num: int) -> str:
    """Handle very large numbers (trillions and beyond)."""
    scale_names = [
        (1000000000000, "trillion"),
        (1000000000000000, "quadrillion"),
        (1000000000000000000, "quintillion"),
        (1000000000000000000000, "sextillion"),
        (1000000000000000000000000, "septillion"),
        (1000000000000000000000000000, "octillion"),
    ]

    for scale, name in reversed(scale_names):
        if num >= scale:
            quotient = num // scale
            remainder = num % scale
            result = f"{int_to_word(quotient)} {name}"
            if remainder > 0:
                result += f" {int_to_word(remainder)}"
            return result

    return str(num)  # Fallback


def word_to_number(word: str) -> Optional[int]:
    """
    Convert word form back to number.

    Args:
        word: Word representation of number

    Returns:
        Integer value or None if conversion fails

    Examples:
        >>> word_to_number("forty-two")
        42
        >>> word_to_number("one hundred twenty-three")
        123
    """
    word = word.lower().strip()

    # Handle negative
    if word.startswith("negative "):
        result = word_to_number(word[9:])
        return -result if result is not None else None

    # Handle zero
    if word == "zero":
        return 0

    # Word to number mapping
    word_to_num = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
        "thousand": 1000,
        "million": 1000000,
        "billion": 1000000000,
        "trillion": 1000000000000,
    }

    # Handle compound words with hyphens
    if "-" in word:
        parts = word.split("-")
        if len(parts) == 2:
            tens_part = word_to_num.get(parts[0], 0)
            ones_part = word_to_num.get(parts[1], 0)
            return tens_part + ones_part

    # Handle multi-word numbers
    words = word.split()
    total = 0
    current = 0

    for w in words:
        if w in word_to_num:
            val = word_to_num[w]
            if val == 100:
                current *= 100
            elif val >= 1000:
                total += current * val
                current = 0
            else:
                current += val
        else:
            return None  # Unknown word

    return total + current


def pluralize(word: str) -> str:
    """
    Enhanced pluralization with more comprehensive rules.

    Args:
        word: Singular form of the word

    Returns:
        Plural form of the word
    """
    word = word.lower().strip()

    # Irregular plurals
    irregular_plurals = {
        "man": "men",
        "woman": "women",
        "child": "children",
        "foot": "feet",
        "tooth": "teeth",
        "mouse": "mice",
        "goose": "geese",
        "person": "people",
        "ox": "oxen",
        "deer": "deer",
        "sheep": "sheep",
        "fish": "fish",
        "moose": "moose",
        "species": "species",
        "series": "series",
        "aircraft": "aircraft",
        "spacecraft": "spacecraft",
    }

    if word in irregular_plurals:
        return irregular_plurals[word]

    # Standard pluralization rules
    if word.endswith(("s", "ss", "sh", "ch", "x", "z")):
        return word + "es"
    elif word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    elif word.endswith(("f", "fe")):
        if word.endswith("fe"):
            return word[:-2] + "ves"
        else:
            return word[:-1] + "ves"
    elif word.endswith("o") and len(word) > 1 and word[-2] not in "aeiou":
        # Some words ending in 'o' add 'es', others just 's'
        es_endings = ["hero", "potato", "tomato", "echo", "embargo", "veto"]
        if word in es_endings:
            return word + "es"
        else:
            return word + "s"
    else:
        return word + "s"


def singularize(word: str) -> str:
    """
    Convert plural form back to singular.

    Args:
        word: Plural form of the word

    Returns:
        Singular form of the word
    """
    word = word.lower().strip()

    # Irregular plurals (reverse mapping)
    irregular_singulars = {
        "men": "man",
        "women": "woman",
        "children": "child",
        "feet": "foot",
        "teeth": "tooth",
        "mice": "mouse",
        "geese": "goose",
        "people": "person",
        "oxen": "ox",
        "deer": "deer",
        "sheep": "sheep",
        "fish": "fish",
        "moose": "moose",
        "species": "species",
        "series": "series",
        "aircraft": "aircraft",
        "spacecraft": "spacecraft",
    }

    if word in irregular_singulars:
        return irregular_singulars[word]

    # Standard singularization rules
    if word.endswith("ies"):
        return word[:-3] + "y"
    elif word.endswith("ves"):
        return word[:-3] + "f"
    elif word.endswith("es"):
        if word.endswith(("ses", "shes", "ches", "xes", "zes")):
            return word[:-2]
        else:
            return word[:-1]
    elif word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    else:
        return word


def extract_numbers_from_text(text: str) -> List[Dict[str, Union[str, int]]]:
    """
    Extract both numeric and word-form numbers from text.

    Args:
        text: Input text

    Returns:
        List of dictionaries with 'word', 'number', and 'position' keys
    """
    results = []

    # Find numeric numbers
    numeric_pattern = r"\b\d+(?:\.\d+)?\b"
    for match in re.finditer(numeric_pattern, text):
        try:
            num = float(match.group()) if "." in match.group() else int(match.group())
            results.append(
                {"word": match.group(), "number": num, "position": match.start()}
            )
        except ValueError:
            continue

    # Find word numbers
    word_numbers = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "trillion",
        "dozen",
        "dozens",
        "couple",
        "few",
        "several",
        "many",
    ]

    for word in word_numbers:
        pattern = r"\b" + re.escape(word) + r"\b"
        for match in re.finditer(pattern, text, re.IGNORECASE):
            num = word_to_number(word)
            if num is not None:
                results.append(
                    {"word": match.group(), "number": num, "position": match.start()}
                )

    # Sort by position in text
    results.sort(key=lambda x: x["position"])
    return results


def format_quantity(count: int, item: str, use_words: bool = True) -> str:
    """
    Format a quantity with proper grammar.

    Args:
        count: Number of items
        item: Item name
        use_words: Whether to use word form for numbers

    Returns:
        Formatted string like "two cats" or "1 cat"
    """
    if count == 0:
        return f"no {pluralize(item)}"
    elif count == 1:
        count_str = "one" if use_words else "1"
        return f"{count_str} {singularize(item)}"
    else:
        count_str = number_to_word(count) if use_words and count <= 100 else str(count)
        return f"{count_str} {pluralize(item)}"


def parse_quantity_phrase(phrase: str) -> Optional[Dict[str, Union[str, int]]]:
    """
    Parse phrases like "three cats" or "a dozen eggs".

    Args:
        phrase: Phrase to parse

    Returns:
        Dictionary with 'count' and 'item' keys, or None if parsing fails
    """
    phrase = phrase.lower().strip()

    # Handle special cases
    special_quantities = {
        "a": 1,
        "an": 1,
        "one": 1,
        "single": 1,
        "couple": 2,
        "pair": 2,
        "both": 2,
        "few": 3,
        "several": 4,
        "many": 10,
        "dozen": 12,
        "dozens": 12,
        "hundred": 100,
        "hundreds": 100,
    }

    words = phrase.split()
    if len(words) < 2:
        return None

    # Try to find quantity word
    quantity_word = words[0]
    item_words = words[1:]

    # Check if first word is a quantity
    if quantity_word in special_quantities:
        count = special_quantities[quantity_word]
    else:
        count = word_to_number(quantity_word)
        if count is None:
            try:
                count = int(quantity_word)
            except ValueError:
                return None

    item = " ".join(item_words)
    return {"count": count, "item": singularize(item)}


def number_to_ordinal(num: int) -> str:
    """Convert number to ordinal form (1st, 2nd, 3rd, etc.)."""
    if num in ORDINAL_WORDS:
        return ORDINAL_WORDS[num]

    if 10 <= num % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(num % 10, "th")  # codespell:ignore.

    return f"{num}{suffix}"


def number_to_ordinal_range(n):
    """
    Convert a number to an ordinal range string.

    Args:
        n (int): The number to convert.

    Returns:
        str: The ordinal range string corresponding to the number.
             For example:
             - 1 -> "first"
             - 2 -> "first to second"
             - 3 -> "first to third"
             If the number is not in the predefined range, the number is returned as a string.
    """
    ordinals = {
        1: "first",
        2: "first to second",
        3: "first to third",
        4: "first to fourth",
        5: "first to fifth",
        6: "first to sixth",
        7: "first to seventh",
        8: "first to eighth",
        9: "first to ninth",
        10: "first to tenth",
        11: "first to eleventh",
        12: "first to twelfth",
        13: "first to thirteenth",
        14: "first to fourteenth",
        15: "first to fifteenth",
        16: "first to sixteenth",
        17: "first to seventeenth",
        18: "first to eighteenth",
        19: "first to nineteenth",
        20: "first to twentieth",
    }
    return ordinals.get(n, str(n))
