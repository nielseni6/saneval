import json
import re
from pathlib import Path
from typing import Dict, List

from ssa.utils.logging import get_log
from ssa.utils.number_to_word import number_to_word, pluralize

log = get_log(__file__)

# Load number words mapping for fallback numeracy extraction
_NUMBER_WORDS_CACHE = None
_PLURAL_MAPPINGS_CACHE = None


def _load_number_words():
    """Load the number words mapping from JSON file."""
    global _NUMBER_WORDS_CACHE
    if _NUMBER_WORDS_CACHE is None:
        try:
            number_words_path = (
                Path(__file__).parent.parent / "data" / "number_words.json"
            )
            with open(number_words_path, "r") as f:
                _NUMBER_WORDS_CACHE = json.load(f)
        except Exception as e:
            log.warning(
                f"Could not load number words mapping: {e}, using basic fallback"
            )
            _NUMBER_WORDS_CACHE = {
                "zero": 0,
                "one": 1,
                "a": 1,
                "an": 1,
                "single": 1,
                "two": 2,
                "couple": 2,
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
                "dozen": 12,
            }
    return _NUMBER_WORDS_CACHE


def _load_plural_mappings():
    """Load plural-to-singular mappings from the SANEval objects file."""
    global _PLURAL_MAPPINGS_CACHE
    if _PLURAL_MAPPINGS_CACHE is None:
        try:
            objects_file_path = (
                Path(__file__).parent.parent
                / "thirdparty"
                / "saneval"
                / "data"
                / "examples"
                / "new_objects.txt"
            )
            plural_to_singular = {}

            with open(objects_file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and " - " in line:
                        singular, plural = line.split(" - ", 1)
                        singular = singular.strip().lower()
                        plural = plural.strip().lower()
                        if singular and plural:
                            plural_to_singular[plural] = singular

            _PLURAL_MAPPINGS_CACHE = plural_to_singular
            log.debug(
                f"Loaded {len(_PLURAL_MAPPINGS_CACHE)} plural-to-singular mappings from SANEval"
            )
        except Exception as e:
            log.error(f"Could not load plural mappings from SANEval: {e}")
            # If SANEval file is unavailable, use empty mappings and rely on regular pluralization rules
            _PLURAL_MAPPINGS_CACHE = {}

    return _PLURAL_MAPPINGS_CACHE


class ObjectAttributeExtractor:
    """
    Extract objects and their attributes from text prompts using LLM.
    """

    def __init__(self, llm):
        """
        Initialize the extractor with an LLM instance.

        Args:
            llm: LLM instance to use for extraction
        """
        self.llm = llm

    def extract(self, prompt: str) -> Dict[str, List[str]]:
        """
        Extract all objects and their attributes from the prompt using LLM.

        Args:
            prompt: Text prompt

        Returns:
            Dict mapping object names to their list of attributes
        """
        llm_prompt = f"""Extract objects and their attributes from the following text prompt. 

Text Prompt: "{prompt}"

Identify all objects (nouns) and their associated attributes (adjectives or descriptive words) in the prompt.

Return the result as a JSON object where:
- Keys are object names (lowercase, singular form)
- Values are lists of attributes associated with each object

For example:
- "a red car and blue bike" would return: {{"car": ["red"], "bike": ["blue"]}}
- "large wooden table" would return: {{"table": ["large", "wooden"]}}
- "small white dog running" would return: {{"dog": ["small", "white"]}}

Only include objects that have explicit attributes mentioned. If an object has no attributes, don't include it.

CRITICAL: Respond with ONLY valid JSON syntax. Do not include any explanations, reasoning, or additional text outside the JSON object."""

        try:
            response = self.llm.call(llm_prompt)

            # Try to parse the JSON response

            # More robust JSON extraction that handles various response formats
            json_str = self._extract_json_from_response(response)
            if json_str:
                object_attributes = json.loads(json_str)

                # Ensure all values are lists and all keys/values are lowercase
                cleaned_attributes = {}
                for obj, attrs in object_attributes.items():
                    obj_name = str(obj).lower().strip()
                    if isinstance(attrs, list):
                        attr_list = [str(attr).lower().strip() for attr in attrs]
                    elif isinstance(attrs, str):
                        attr_list = [str(attrs).lower().strip()]
                    else:
                        attr_list = []

                    if obj_name and attr_list:
                        cleaned_attributes[obj_name] = attr_list

                log.debug(f"LLM extracted objects and attributes: {cleaned_attributes}")
                return cleaned_attributes

            else:
                log.warning(f"Could not extract JSON from LLM response: {response}")
                return {}

        except json.JSONDecodeError as e:
            log.error(f"Failed to parse JSON from LLM response: {e}")
            log.debug(f"Raw LLM response: {response}")
            return {}
        except Exception as e:
            log.error(f"Error extracting object attributes with LLM: {e}")
            return {}

    def extract_numeracy_objects(self, prompt: str) -> Dict[str, int]:
        """
        Extract objects and their quantities from the prompt using LLM.

        Args:
            prompt: Text prompt

        Returns:
            Dict mapping object names to their quantities
        """
        llm_prompt = f"""Extract all objects and their quantities from the following text prompt.

Text Prompt: "{prompt}"

Identify all objects (nouns) and their associated quantities (numbers) mentioned in the prompt. Include both written numbers (like "seven", "three") and numeric digits (like "7", "3").

Return the result as a JSON object where:
- Keys are object names (lowercase, singular form)
- Values are the quantities as integers

For example:
- "seven horses and 4 pigs" would return: {{"horse": 7, "pig": 4}}
- "three red cars and two blue bikes" would return: {{"car": 3, "bike": 2}}
- "a single dog and five cats" would return: {{"dog": 1, "cat": 5}}
- "ten books on the shelf" would return: {{"book": 10, "shelf": 1}}
- "one woman wearing two shoes" would return: {{"woman": 1, "shoe": 2}}
- "dozens of apples and hundreds of oranges" would return: {{"apple": 12, "orange": 100}}
- "a couple of birds and a few fish" would return: {{"bird": 2, "fish": 3}}

Convert written numbers to integers:
- "one"/"a"/"single" = 1
- "two"/"couple" = 2  
- "three" = 3
- "four" = 4
- "half a dozen" = 6
- "dozen" = 12
- "hundred" = 100

Only include objects that have explicit quantities mentioned. Convert object names to singular form.

CRITICAL: Respond with ONLY valid JSON syntax. Do not include any explanations, reasoning, or additional text outside the JSON object."""

        try:
            response = self.llm.call(llm_prompt)

            # More robust JSON extraction that handles various response formats
            json_str = self._extract_json_from_response(response)
            if json_str:
                numeracy_objects = json.loads(json_str)

                # Ensure all keys are strings and values are integers
                cleaned_numeracy = {}
                for obj, quantity in numeracy_objects.items():
                    obj_name = str(obj).lower().strip()
                    try:
                        quantity_int = int(quantity)
                        if obj_name and quantity_int > 0:
                            cleaned_numeracy[obj_name] = quantity_int
                    except (ValueError, TypeError):
                        log.warning(
                            f"Could not convert quantity '{quantity}' to integer for object '{obj_name}'"
                        )

                log.debug(f"LLM extracted numeracy objects: {cleaned_numeracy}")
                return cleaned_numeracy

            else:
                log.warning(f"Could not extract JSON from LLM response: {response}")
                return {}

        except json.JSONDecodeError as e:
            log.error(f"Failed to parse JSON from LLM response: {e}")
            log.debug(f"Raw LLM response: {response}")
            return {}
        except Exception as e:
            log.error(f"Error extracting numeracy objects with LLM: {e}")
            return {}

    def extract_objects(self, prompt: str) -> List[str]:
        """
        Extract only the objects (nouns) from the prompt using LLM, without considering attributes.

        Args:
            prompt: Text prompt

        Returns:
            List of object names (lowercase, singular form)
        """
        llm_prompt = f"""Extract all significant objects (nouns) from the following text prompt.

Text Prompt: "{prompt}"

Identify all significant objects (nouns) mentioned in the prompt, regardless of any attributes they may have. Focus on:
- Living beings (people, animals, creatures)
- Substantial physical objects (vehicles, furniture, buildings, tools, toys)
- Natural features (trees, rocks, mountains, water bodies)
- Important branded items or recognizable objects

Do NOT include:
- Positional/directional terms (top, bottom, left, right, front, back, side, center)
- Clothing items (shirts, pants, shoes, hats) unless they are the main focus
- Body parts (hands, face, legs) unless they are the main focus
- Abstract concepts (time, love, happiness)
- Very small or insignificant items (buttons, zippers, laces)
- Prepositions or spatial relationships (on, in, under, above, below)

Return the result as a JSON array containing object names in lowercase, singular form.

For example:
- "a red car and blue bike" would return: ["car", "bike"]
- "large wooden table with metal legs" would return: ["table"]
- "small white dog running in the park" would return: ["dog", "park"]
- "three books on the shelf" would return: ["book", "shelf"]
- "John's iPhone on the Tesla dashboard" would return: ["phone", "dashboard"]
- "McDonald's burger with Coca-Cola" would return: ["burger", "soda"]
- "Paris street with Eiffel Tower and Jennifer Lawrence" would return: ["street", "eiffel tower", "person"]
- "woman wearing Nike shoes and Rolex watch" would return: ["woman", "shoes", "watch"]
- "girl on the top of a frog" would return: ["girl", "frog"]
- "cat under the table" would return: ["cat", "table"]

Include all significant objects mentioned, even if they don't have explicit attributes. Convert proper nouns to lowercase but preserve compound names (like "eiffel tower").

CRITICAL: Respond with ONLY valid JSON syntax. Do not include any explanations, reasoning, or additional text outside the JSON array."""

        try:
            response = self.llm.call(llm_prompt)

            # More robust JSON extraction that handles various response formats
            json_str = self._extract_json_from_response(response)
            if json_str:
                objects_list = json.loads(json_str)

                # Ensure all items are strings and convert to lowercase
                cleaned_objects = []
                for obj in objects_list:
                    obj_name = str(obj).lower().strip()
                    if (
                        obj_name and obj_name not in cleaned_objects
                    ):  # Remove duplicates
                        cleaned_objects.append(obj_name)

                log.debug(f"LLM extracted objects: {cleaned_objects}")
                return cleaned_objects

            else:
                log.warning(f"Could not extract JSON from LLM response: {response}")
                return []

        except json.JSONDecodeError as e:
            log.error(f"Failed to parse JSON from LLM response: {e}")
            log.debug(f"Raw LLM response: {response}")
            return []
        except Exception as e:
            log.error(f"Error extracting objects with LLM: {e}")
            return []

    def numeric_diff(
        self, detected_objects: List[str], expected_objects: Dict[str, int]
    ) -> List[str]:
        """
        Compare detected objects with expected quantities and return missing items.
        Uses LLM to handle synonyms and merge synonymous entries.

        Args:
            detected_objects: List of detected object names (can contain duplicates)
            expected_objects: Dict mapping object names to their expected quantities

        Returns:
            List of missing items in format ["two pigs", "four horses"]
        """
        # Count occurrences of each detected object
        detected_counts = {}
        for obj in detected_objects:
            obj_name = str(obj).lower().strip()
            detected_counts[obj_name] = detected_counts.get(obj_name, 0) + 1

        # Get all unique object names from both detected and expected
        all_objects = set(detected_counts.keys()) | set(expected_objects.keys())

        # Get unified synonym mapping for all objects
        unified_synonyms = self._get_unified_synonyms(list(all_objects))

        # Merge both dictionaries using the same synonym mapping
        merged_detected = self._apply_synonym_mapping(detected_counts, unified_synonyms)
        merged_expected = self._apply_synonym_mapping(
            expected_objects, unified_synonyms
        )

        # Find missing objects and quantities
        missing_items = []

        for expected_obj, expected_count in merged_expected.items():
            expected_obj_lower = expected_obj.lower().strip()
            detected_count = merged_detected.get(expected_obj_lower, 0)

            if detected_count < expected_count:
                missing_count = expected_count - detected_count

                # Convert count to word form for better readability
                count_word = number_to_word(missing_count)

                # Handle singular vs plural
                if missing_count == 1:
                    obj_name = expected_obj_lower
                else:
                    obj_name = pluralize(expected_obj_lower)

                missing_items.append(f"{count_word} {obj_name} is/are missing")

            elif detected_count > expected_count:
                extra_count = detected_count - expected_count

                count_word = number_to_word(extra_count)

                if extra_count == 1:
                    obj_name = expected_obj_lower
                else:
                    obj_name = pluralize(expected_obj_lower)

                missing_items.append(f"extra {count_word} {obj_name} detected")

        log.debug(f"Original detected counts: {detected_counts}")
        log.debug(f"Merged detected counts: {merged_detected}")
        log.debug(f"Original expected counts: {expected_objects}")
        log.debug(f"Merged expected counts: {merged_expected}")
        log.debug(f"Missing items: {missing_items}")

        return missing_items

    def object_diff(
        self, detected_objects: List[str], expected_objects: List[str]
    ) -> List[str]:
        """
        Compare detected objects with expected objects and return missing object types.
        Uses LLM to handle synonyms and merge synonymous entries.
        Does not consider quantities - only checks if object types are present.

        Args:
            detected_objects: List of detected object names (can contain duplicates)
            expected_objects: List of expected object names (can contain duplicates)

        Returns:
            List of missing object types (unique object names in expected but not in detected after synonym mapping)
        """
        # Get unique object names from both lists
        detected_unique = set()
        for obj in detected_objects:
            obj_name = str(obj).lower().strip()
            if obj_name:
                detected_unique.add(obj_name)

        expected_unique = set()
        for obj in expected_objects:
            obj_name = str(obj).lower().strip()
            if obj_name:
                expected_unique.add(obj_name)

        # Get all unique object names from both detected and expected
        all_objects = detected_unique | expected_unique

        # Get unified synonym mapping for all objects
        unified_synonyms = self._get_unified_synonyms(list(all_objects))

        # Convert sets to presence dictionaries (1 if present, 0 if not)
        detected_presence = {obj: 1 for obj in detected_unique}
        expected_presence = {obj: 1 for obj in expected_unique}

        # Apply synonym mapping to both sets
        merged_detected = self._apply_synonym_mapping(
            detected_presence, unified_synonyms
        )
        merged_expected = self._apply_synonym_mapping(
            expected_presence, unified_synonyms
        )

        # Find missing object types
        missing_objects = []

        for expected_obj in merged_expected.keys():
            expected_obj_lower = expected_obj.lower().strip()
            if expected_obj_lower not in merged_detected:
                completed_obj = f"The missing object: {expected_obj_lower}"
                missing_objects.append(completed_obj)

        log.debug(f"Original detected objects: {detected_unique}")
        log.debug(f"Merged detected objects: {set(merged_detected.keys())}")
        log.debug(f"Original expected objects: {expected_unique}")
        log.debug(f"Merged expected objects: {set(merged_expected.keys())}")
        log.debug(f"Missing object types: {missing_objects}")

        return missing_objects

    def normalize_detected_objects(
        self, prompt: str, detected_objects: List[str]
    ) -> List[str]:
        """
        Normalize detected object names using expected objects from the prompt as reference.
        Uses LLM to handle synonyms and map detected objects to expected vocabulary.

        Args:
            prompt: Text prompt containing expected objects
            detected_objects: List of detected object names that may contain synonyms

        Returns:
            List of detected objects with names normalized to match expected vocabulary
        """
        # Extract expected objects from the prompt
        expected_objects = self.extract_objects(prompt)

        if not expected_objects or not detected_objects:
            return detected_objects

        # Get all unique object names from both detected and expected
        detected_unique = set()
        for obj in detected_objects:
            obj_name = str(obj).lower().strip()
            if obj_name:
                detected_unique.add(obj_name)

        expected_unique = set()
        for obj in expected_objects:
            obj_name = str(obj).lower().strip()
            if obj_name:
                expected_unique.add(obj_name)

        all_objects = detected_unique | expected_unique

        # Get unified synonym mapping for all objects
        unified_synonyms = self._get_unified_synonyms(list(all_objects))

        # Ensure that keys in synonym mapping are always from expected objects (prompt)
        # If a key is not from expected objects but values contain expected objects, swap them
        corrected_synonyms = {}
        for standard_term, synonyms in unified_synonyms.items():
            # Check if the current key is from expected objects
            if standard_term in expected_unique:
                # Key is already from expected objects, keep as is
                corrected_synonyms[standard_term] = synonyms
            else:
                # Key is not from expected objects, check if any synonym is from expected
                expected_synonym = None
                for synonym in synonyms:
                    if synonym in expected_unique:
                        expected_synonym = synonym
                        break

                if expected_synonym:
                    # Swap: make the expected object the key and include the original key in values
                    new_synonyms = synonyms.copy()
                    if standard_term not in new_synonyms:
                        new_synonyms.append(standard_term)
                    corrected_synonyms[expected_synonym] = new_synonyms
                else:
                    # No expected objects in this group, keep original mapping
                    corrected_synonyms[standard_term] = synonyms

        # Use corrected synonym mapping
        unified_synonyms = corrected_synonyms

        # Create a reverse mapping from detected objects to expected standard terms
        detected_to_expected = {}

        for standard_term, synonyms in unified_synonyms.items():
            # If the standard term is in expected objects, use it as the target
            if standard_term in expected_unique:
                for synonym in synonyms:
                    if synonym in detected_unique:
                        detected_to_expected[synonym] = standard_term
            else:
                # If standard term is not in expected, find if any synonym is in expected
                expected_synonym = None
                for synonym in synonyms:
                    if synonym in expected_unique:
                        expected_synonym = synonym
                        break

                if expected_synonym:
                    for synonym in synonyms:
                        if synonym in detected_unique:
                            detected_to_expected[synonym] = expected_synonym

        # Normalize the detected objects list
        normalized_objects = []
        for obj in detected_objects:
            obj_name = str(obj).lower().strip()
            if obj_name in detected_to_expected:
                normalized_objects.append(detected_to_expected[obj_name])
            else:
                normalized_objects.append(obj_name)

        log.debug(f"Original detected objects: {detected_objects}")
        log.debug(f"Expected objects from prompt: {expected_objects}")
        # log.debug(f"Original synonym mapping: {self._get_unified_synonyms(list(all_objects))}")
        log.debug(f"Corrected synonym mapping (keys from prompt): {unified_synonyms}")
        log.debug(f"Detected to expected mapping: {detected_to_expected}")
        log.debug(f"Normalized detected objects: {normalized_objects}")

        return normalized_objects

    def _get_unified_synonyms(self, object_names: List[str]) -> Dict[str, List[str]]:
        """
        Get unified synonym mapping for all objects using LLM.

        Args:
            object_names: List of all object names to analyze

        Returns:
            Dictionary mapping standard terms to lists of synonyms
        """
        if len(object_names) < 2:
            return {obj: [obj] for obj in object_names}

        llm_prompt = f"""Analyze the following list of objects and identify any synonyms or similar objects that should be grouped together.

Objects: {object_names}

For each group of synonymous objects, choose the most common/standard term to represent the group.

Return the result as a JSON object where:
- Keys are the standard/representative terms (lowercase, singular form)
- Values are arrays of all synonymous terms that should be grouped under that key

For example:
- ["boy", "child", "person", "girl"] might return: {{"person": ["boy", "child", "person", "girl"]}}
- ["house", "home", "building"] might return: {{"house": ["house", "home", "building"]}}
- ["car", "vehicle", "automobile"] might return: {{"car": ["car", "vehicle", "automobile"]}}
- ["dog", "cat", "bird"] might return: {{"dog": ["dog"], "cat": ["cat"], "bird": ["bird"]}}
- ["bee", "bird", "animal"] might return {{"bee": ["bee"], "bird": ["bird", "animal"]}}
- ["desk", "table", "furniture"] might return {{"desk": ["desk", "table"], "furniture": ["furniture"]}}
Only group objects that are truly synonymous or refer to the same type of thing. Keep unrelated objects separate.

CRITICAL: Respond with ONLY valid JSON syntax. Do not include any explanations, reasoning, or additional text outside the JSON object."""

        try:
            response = self.llm.call(llm_prompt)

            # More robust JSON extraction that handles various response formats
            json_str = self._extract_json_from_response(response)
            if json_str:
                synonym_groups = json.loads(json_str)

                # Normalize the synonym groups
                normalized_groups = {}
                processed_objects = set()

                for standard_term, synonyms in synonym_groups.items():
                    standard_term = standard_term.lower().strip()
                    normalized_synonyms = []

                    for synonym in synonyms:
                        synonym = synonym.lower().strip()
                        if synonym in object_names:
                            normalized_synonyms.append(synonym)
                            processed_objects.add(synonym)

                    if normalized_synonyms:
                        normalized_groups[standard_term] = normalized_synonyms

                # Add any objects that weren't grouped
                for obj_name in object_names:
                    obj_name = obj_name.lower().strip()
                    if obj_name not in processed_objects:
                        normalized_groups[obj_name] = [obj_name]

                log.debug(f"Unified synonym groups: {normalized_groups}")
                return normalized_groups

            else:
                log.warning(
                    f"Could not extract JSON from LLM response for unified synonym detection: {response}"
                )
                return {obj: [obj] for obj in object_names}

        except json.JSONDecodeError as e:
            log.error(
                f"Failed to parse JSON from LLM response for unified synonym detection: {e}"
            )
            log.debug(f"Raw LLM response: {response}")
            return {obj: [obj] for obj in object_names}
        except Exception as e:
            log.error(f"Error detecting unified synonyms with LLM: {e}")
            return {obj: [obj] for obj in object_names}

    def _apply_synonym_mapping(
        self, objects_dict: Dict[str, int], synonym_mapping: Dict[str, List[str]]
    ) -> Dict[str, int]:
        """
        Apply synonym mapping to merge counts for synonymous objects.

        Args:
            objects_dict: Dictionary mapping object names to their counts
            synonym_mapping: Dictionary mapping standard terms to lists of synonyms

        Returns:
            Dictionary with synonymous objects merged using the provided mapping
        """
        merged_dict = {}

        for standard_term, synonyms in synonym_mapping.items():
            total_count = 0

            # Sum up counts for all synonyms
            for synonym in synonyms:
                synonym = synonym.lower().strip()
                if synonym in objects_dict:
                    total_count += objects_dict[synonym]

            if total_count > 0:
                merged_dict[standard_term] = total_count

        log.debug(f"Applied synonym mapping: {objects_dict} -> {merged_dict}")
        return merged_dict

    def _fix_json_syntax_errors(self, json_str: str) -> str:
        """
        Fix common JSON syntax errors that LLMs might generate.

        Args:
            json_str: JSON string that may contain syntax errors

        Returns:
            Fixed JSON string
        """
        # First check if the JSON is already valid
        try:
            json.loads(json_str)
            log.debug(f"JSON is already valid, no fixes needed: {json_str}")
            return json_str
        except json.JSONDecodeError:
            # JSON needs fixing, continue with processing
            pass

        fixed_str = json_str

        # First, remove any extra text after the final closing brace
        # "toy": 6} extra text here -> "toy": 6}
        json_match = re.search(r"^(.*\})\s+.*$", fixed_str, re.DOTALL)
        if json_match:
            potential_trim = json_match.group(1).strip()
            if potential_trim.endswith("}"):
                try:
                    # Test if the trimmed version is valid JSON
                    json.loads(potential_trim)
                    fixed_str = potential_trim
                except json.JSONDecodeError:
                    # If trimmed version isn't valid, keep processing the original
                    pass

        # CRITICAL: Fix the specific "haughty" pattern FIRST before any comma processing
        # This prevents the comma-fixing logic from incorrectly merging structures
        # Pattern: "airplane": 6\nhaughty },\n"numbers_found" -> "airplane": 6\n},\n"numbers_found"
        if "haughty" in fixed_str and '"objects"' in fixed_str:
            fixed_str = re.sub(
                r'("?\w+"?\s*:\s*\d+)\s*\n\s*haughty\s*\}',
                r"\1\n}",
                fixed_str,
            )

        # Fix malformed object values with trailing text after numbers
        # "pig": 1 accelerators -> "pig": 1
        # Handle both same-line and newline cases
        fixed_str = re.sub(
            r'("?\w+"?\s*:\s*\d+)\s*\n\s*[a-zA-Z]+(?:\s+[a-zA-Z]+)*',
            r"\1",
            fixed_str,
        )
        fixed_str = re.sub(
            r'("?\w+"?\s*:\s*\d+)\s+[a-zA-Z]+(?:\s+[a-zA-Z]+)*',
            r"\1",
            fixed_str,
        )

        # Fix missing commas between key-value pairs at the same nesting level
        # "pig": 1\n  "numbers_found" -> "pig": 1,\n  "numbers_found"
        # But avoid adding commas before closing braces
        # Apply this multiple times to handle consecutive missing commas
        for _ in range(5):  # Maximum 5 iterations to avoid infinite loops
            new_fixed_str = re.sub(
                r'("?\w+"?\s*:\s*(?:\d+|\[[^\]]*\]|"[^"]*"))\s*\n\s*("?\w+"?\s*:)',
                r"\1,\n  \2",
                fixed_str,
            )
            # Don't add comma if the next line is a closing brace
            new_fixed_str = re.sub(
                r'("?\w+"?\s*:\s*(?:\d+|\[[^\]]*\]|"[^"]*")),\s*\n\s*\}',
                r"\1\n  }",
                new_fixed_str,
            )
            if new_fixed_str == fixed_str:
                break  # No more changes, stop iterating
            fixed_str = new_fixed_str

        # Fix cases where there's a closing brace followed by comma and more content
        # "motorcycle": 1 }, -> "motorcycle": 1,
        fixed_str = re.sub(
            r'("?\w+"?\s*:\s*\d+)\s*\},',
            r"\1,",
            fixed_str,
        )

        # The original patterns - keep these for backwards compatibility
        # Fix the specific case: missing closing brace after object definition
        # "motorcycle": 1\n  ,\n  "numbers_found" -> "motorcycle": 1,\n  "numbers_found"
        fixed_str = re.sub(
            r'("?\w+"?\s*:\s*\d+)\s*\n\s*,\s*\n\s*("(?:numbers_found|objects)")',
            r"\1,\n  \2",
            fixed_str,
        )

        # Final check: if there's still extra text after all fixes, try trimming again
        if not fixed_str.strip().endswith("}"):
            last_brace_match = re.search(r"^(.*\})", fixed_str, re.DOTALL)
            if last_brace_match:
                potential_trim = last_brace_match.group(1).strip()
                try:
                    json.loads(potential_trim)
                    fixed_str = potential_trim
                except json.JSONDecodeError:
                    pass

        log.debug(f"Fixed JSON syntax: {json_str} -> {fixed_str}")
        return fixed_str

    def _singularize(self, word: str) -> str:
        """
        Convert plural words to singular form using SANEval mappings.

        Args:
            word: Word to singularize

        Returns:
            Singular form of the word
        """
        word = word.lower().strip()

        # First check the SANEval plural mappings
        plural_mappings = _load_plural_mappings()
        if word in plural_mappings:
            return plural_mappings[word]

        # Handle regular plurals for words not in SANEval
        if len(word) > 3 and word.endswith("s"):
            # Don't singularize words that end in 'ss', 'us', 'is'
            if not word.endswith(("ss", "us", "is")):
                # Handle -ies -> -y
                if word.endswith("ies") and len(word) > 4:
                    return word[:-3] + "y"
                # Handle -es endings (but be careful not to over-singularize)
                elif word.endswith("es") and len(word) > 4:
                    # Only remove -es for certain patterns (but not for words ending in -rses like horses)
                    if word.endswith(
                        ("sses", "ches", "shes", "xes", "zes")
                    ) and not word.endswith("rses"):
                        return word[:-2]
                    else:
                        # For words like "horses", just remove the -s
                        return word[:-1]
                # Handle regular -s endings
                else:
                    return word[:-1]

        return word

    def _parse_compound_number(self, text: str) -> int:
        """
        Parse compound numbers like 'twenty six', 'thirty two', etc.

        Args:
            text: Text containing potential compound numbers

        Returns:
            Integer value of the compound number, or 0 if not found
        """
        number_words = _load_number_words()

        # Common pattern: "twenty six", "thirty two", etc.
        compound_pattern = r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(one|two|three|four|five|six|seven|eight|nine)\b"

        match = re.search(compound_pattern, text.lower())
        if match:
            tens_word, ones_word = match.groups()
            tens_value = number_words.get(tens_word, 0)
            ones_value = number_words.get(ones_word, 0)
            return tens_value + ones_value

        return 0

    def _fallback_numeracy_extraction(
        self, prompt: str, available_numbers: List[str]
    ) -> Dict[str, any]:
        """
        Fallback numeracy extraction using regex pattern matching and number words mapping.
        Supports compound numbers like 'twenty six', 'thirty two', etc.

        Args:
            prompt: Text prompt containing numeracy relationships
            available_numbers: List of valid number words (from NUMERIC_QUERIES)

        Returns:
            Dictionary with 'objects' (Dict[str, int]) and 'numbers_found' (List[str]) keys
        """
        number_words = _load_number_words()

        # Extract all number words and digits from the prompt
        numbers_found = set()  # Use set to avoid duplicates
        objects_dict = {}

        # Common stop words and prepositions to filter out
        stop_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "a",
            "an",
            "this",
            "that",
            "these",
            "those",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "can",
            "may",
            "must",
            "shall",
            "should",
            "will",
            "would",
        }

        # Color and adjective words that are not objects
        adjectives = {
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "brown",
            "purple",
            "orange",
            "pink",
            "gray",
            "grey",
            "big",
            "small",
            "large",
            "tiny",
            "huge",
            "little",
            "old",
            "new",
            "young",
            "tall",
            "short",
            "long",
            "wide",
            "narrow",
            "thick",
            "thin",
            "heavy",
            "light",
            "fast",
            "slow",
            "single",
            "multiple",
            "many",
            "few",
            "several",
        }

        # FIRST: Handle compound numbers like "twenty six cars", "thirty two dogs"
        compound_patterns = [
            # Compound number + of + object: "twenty six of cats"
            r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(one|two|three|four|five|six|seven|eight|nine)\s+of\s+(\w+s?)\b",
            # Compound number + adjective + object: "twenty six red cars"
            r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(one|two|three|four|five|six|seven|eight|nine)\s+(?:"
            + "|".join(adjectives)
            + r")\s+(\w+s?)\b",
            # Compound number + object: "twenty six cats", "thirty two dogs"
            r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(one|two|three|four|five|six|seven|eight|nine)\s+(\w+s?)\b",
        ]

        for pattern in compound_patterns:
            matches = re.findall(pattern, prompt.lower(), re.IGNORECASE)
            for match in matches:
                tens_word, ones_word, obj = match[0], match[1], match[2]

                # Clean the object name
                obj_clean = obj.lower().strip()

                # Filter out stop words, adjectives, and short words
                if (
                    obj_clean in stop_words
                    or obj_clean in adjectives
                    or len(obj_clean) <= 2
                    or not obj_clean.isalpha()
                ):
                    continue

                # Calculate compound number value
                tens_value = number_words.get(tens_word, 0)
                ones_value = number_words.get(ones_word, 0)
                quantity = tens_value + ones_value

                # Record the compound number found
                compound_number = f"{tens_word} {ones_word}"
                numbers_found.add(compound_number)

                # Handle pluralization - convert to singular form
                obj_clean = self._singularize(obj_clean)

                # Store the object and quantity (prioritize first match)
                if obj_clean and obj_clean not in objects_dict and quantity > 0:
                    objects_dict[obj_clean] = quantity

        # SECOND: Handle regular single-word patterns (always process to catch mixed scenarios)
        # But skip words that were part of compound numbers
        compound_number_words = set()
        for pattern in compound_patterns:
            matches = re.findall(pattern, prompt.lower(), re.IGNORECASE)
            for match in matches:
                tens_word, ones_word = match[0], match[1]
                compound_number_words.add(tens_word)
                compound_number_words.add(ones_word)

        # Prioritized patterns - more specific patterns first to avoid mismatches
        patterns = [
            # Number + of + object: "couple of birds", "dozen of apples" (most specific)
            (
                r"\b(?:(\d+)|("
                + "|".join(re.escape(word) for word in number_words.keys())
                + r"))\s+of\s+(\w+s?)\b",
                3,
            ),
            # Number + adjective + object: "three red cars", "two blue bikes" (specific to adjective + noun)
            (
                r"\b(?:(\d+)|("
                + "|".join(re.escape(word) for word in number_words.keys())
                + r"))\s+(?:"
                + "|".join(adjectives)
                + r")\s+(\w+s?)\b",
                3,
            ),
            # Direct number + object: "seven women", "3 cats" (general pattern)
            (
                r"\b(?:(\d+)|("
                + "|".join(re.escape(word) for word in number_words.keys())
                + r"))\s+(\w+s?)\b",
                3,
            ),
        ]

        # Process patterns in order of specificity
        for pattern, obj_group in patterns:
            matches = re.findall(pattern, prompt.lower(), re.IGNORECASE)

            for match in matches:
                digit, word, obj = match[0], match[1], match[obj_group - 1]

                # Skip if this word was part of a compound number
                if word and word.lower() in compound_number_words:
                    continue

                # Clean the object name
                obj_clean = obj.lower().strip()

                # Filter out stop words, adjectives, and short words
                if (
                    obj_clean in stop_words
                    or obj_clean in adjectives
                    or len(obj_clean) <= 2
                    or not obj_clean.isalpha()
                ):
                    continue

                # Determine the quantity
                if digit:
                    quantity = int(digit)
                    numbers_found.add(digit)
                elif word:
                    quantity = int(number_words.get(word.lower(), 1))
                    numbers_found.add(word.lower())
                else:
                    continue

                # Handle pluralization - convert to singular form
                obj_clean = self._singularize(obj_clean)

                # Store the object and quantity (prioritize first match)
                if obj_clean and obj_clean not in objects_dict:
                    objects_dict[obj_clean] = quantity

        # THIRD: If still no matches with patterns, try simpler word-based approach
        if not objects_dict:
            # Try compound number parsing first
            compound_value = self._parse_compound_number(prompt)
            if compound_value > 0:
                # Find potential object words (nouns)
                words = prompt.lower().split()
                potential_objects = []

                for word in words:
                    word = word.strip('.,!?;:"()[]{}')
                    if (
                        word.isalpha()
                        and len(word) > 2
                        and word not in stop_words
                        and word not in adjectives
                        and word not in number_words
                    ):
                        potential_objects.append(word)

                if potential_objects:
                    obj = potential_objects[0]
                    # Record compound number found (estimate from compound_value)
                    numbers_found.add(f"compound_{compound_value}")

                    # Clean object name
                    obj_clean = self._singularize(obj.lower().strip())

                    objects_dict[obj_clean] = compound_value
            else:
                # Fall back to regular single-word number approach
                all_number_pattern = (
                    r"\b("
                    + "|".join(re.escape(word) for word in number_words.keys())
                    + r"|\d+)\b"
                )
                found_numbers = re.findall(
                    all_number_pattern, prompt.lower(), re.IGNORECASE
                )

                # Find potential object words (nouns)
                words = prompt.lower().split()
                potential_objects = []

                for word in words:
                    word = word.strip('.,!?;:"()[]{}')
                    if (
                        word.isalpha()
                        and len(word) > 2
                        and word not in stop_words
                        and word not in adjectives
                        and word not in number_words
                    ):
                        potential_objects.append(word)

                # Try to pair first number with first potential object
                if found_numbers and potential_objects:
                    num_str = found_numbers[0]
                    obj = potential_objects[0]

                    if num_str.isdigit():
                        quantity = int(num_str)
                        numbers_found.add(num_str)
                    else:
                        quantity = int(number_words.get(num_str.lower(), 1))
                        numbers_found.add(num_str.lower())

                    # Clean object name
                    obj_clean = self._singularize(obj.lower().strip())

                    objects_dict[obj_clean] = quantity

        result = {
            "objects": objects_dict,
            "numbers_found": list(numbers_found),  # Convert back to list
        }

        log.debug(f"Fallback numeracy extraction result: {result}")
        return result

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from LLM response that may contain extra text, explanations, or formatting.

        Args:
            response: Raw LLM response that may contain JSON mixed with other text

        Returns:
            Extracted JSON string, or None if no valid JSON found
        """
        if not response or not response.strip():
            return None

        # Remove markdown code blocks if present
        response = re.sub(r"```json\s*", "", response)
        response = re.sub(r"```\s*$", "", response)

        # Try multiple JSON extraction strategies

        # Strategy 1: Look for complete JSON objects (most common case)
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
        if json_match:
            potential_json = json_match.group()
            try:
                # Test if it's valid JSON
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass

        # Strategy 2: Look for JSON objects with nested structures
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            potential_json = json_match.group()

            # Clean up common issues in the extracted JSON
            # Remove any text that appears after the JSON structure
            lines = potential_json.split("\n")
            clean_lines = []
            in_json = False
            brace_count = 0

            for line in lines:
                for char in line:
                    if char == "{":
                        brace_count += 1
                        in_json = True
                    elif char == "}":
                        brace_count -= 1

                if in_json:
                    clean_lines.append(line)

                # Stop when we've closed all braces
                if in_json and brace_count == 0:
                    break

            if clean_lines:
                cleaned_json = "\n".join(clean_lines)
                try:
                    # Test if it's valid JSON after cleaning
                    json.loads(cleaned_json)
                    return cleaned_json
                except json.JSONDecodeError:
                    # Return original if cleaning didn't help
                    return potential_json

            return potential_json

        # Strategy 3: Look for JSON arrays (for methods that return lists)
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            potential_json = json_match.group()
            try:
                # Test if it's valid JSON
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass

        log.warning(f"Could not extract valid JSON from LLM response: {response}")
        return None

    def extract_numeracy_relationship(
        self, prompt: str, available_numbers: List[str]
    ) -> Dict[str, any]:
        """
        Extract numeracy relationship with objects and their quantities from the prompt using LLM.

        Args:
            prompt: Text prompt containing numeracy relationships
            available_numbers: List of valid number words (from NUMERIC_QUERIES)

        Returns:
            Dictionary with 'objects' (Dict[str, int]) and 'numbers_found' (List[str]) keys,
            or empty dict if extraction fails
        """
        llm_prompt = f"""Extract all objects and their exact quantities from the following text prompt.

Text Prompt: "{prompt}"

Available number words: {available_numbers}

Identify all objects (nouns) and their associated quantities (numbers) mentioned in the prompt. Include both written numbers (like "seven", "three") and numeric digits (like "7", "3").

Return the result as a JSON object with exactly these keys:
- "objects": a dictionary mapping object names to their quantities as integers
- "numbers_found": a list of the number words/digits found in the prompt

For example:
- "seven horses and 4 pigs" would return: {{"objects": {{"horse": 7, "pig": 4}}, "numbers_found": ["seven", "4"]}}
- "three red cars and two blue bikes" would return: {{"objects": {{"car": 3, "bike": 2}}, "numbers_found": ["three", "two"]}}
- "a single dog and five cats" would return: {{"objects": {{"dog": 1, "cat": 5}}, "numbers_found": ["a", "five"]}}
- "ten books on the shelf" would return: {{"objects": {{"book": 10, "shelf": 1}}, "numbers_found": ["ten"]}}

Important conversion rules:
- "one"/"a"/"an"/"single" = 1
- "two"/"couple" = 2
- "three" = 3
- "four" = 4
- "five" = 5
- "six" = 6
- "seven" = 7
- "eight" = 8
- "nine" = 9
- "ten" = 10
- "dozen" = 12
- "hundred" = 100

Guidelines:
- Convert object names to lowercase, singular form
- Only include objects that have explicit or implied quantities
- If no specific number is mentioned for an object but it appears in a counting context, assume quantity 1
- Focus on the main objects being counted in the prompt

CRITICAL: Respond with ONLY valid JSON syntax. Do not include any explanations, reasoning, or additional text outside the JSON object."""

        try:
            response = self.llm.call(llm_prompt)

            # More robust JSON extraction that handles various response formats
            json_str = self._extract_json_from_response(response)
            if json_str:
                # Fix common JSON syntax errors before parsing
                json_str = self._fix_json_syntax_errors(json_str)

                numeracy_data = json.loads(json_str)

                # Validate the response structure
                if not isinstance(numeracy_data, dict):
                    log.warning(f"LLM response is not a dictionary: {numeracy_data}")
                    return {}

                # Check for empty response (valid case when no numeracy relationship found)
                if not numeracy_data:
                    log.debug("LLM found no numeracy relationship in prompt")
                    return {}

                # Handle case where LLM returns objects directly at root level instead of nested under "objects"
                # This commonly happens when LLM returns {"woman": 7, "numbers_found": ["seven"]}
                # instead of {"objects": {"woman": 7}, "numbers_found": ["seven"]}
                if "numbers_found" in numeracy_data and "objects" not in numeracy_data:
                    # Extract numbers_found and treat everything else as objects
                    numbers_found = numeracy_data.pop("numbers_found")
                    objects = {
                        k: v
                        for k, v in numeracy_data.items()
                        if isinstance(v, (int, float, str))
                    }
                    numeracy_data = {"objects": objects, "numbers_found": numbers_found}
                    log.debug(
                        f"Restructured LLM response to expected format: {numeracy_data}"
                    )

                # Validate required keys
                required_keys = ["objects", "numbers_found"]
                if not all(key in numeracy_data for key in required_keys):
                    log.warning(
                        f"Missing required keys in LLM response. Got: {numeracy_data.keys()}, Expected: {required_keys}"
                    )
                    return {}

                # Clean and validate the extracted data
                objects = numeracy_data.get("objects", {})
                numbers_found = numeracy_data.get("numbers_found", [])

                # Validate objects dictionary
                if not isinstance(objects, dict):
                    log.warning(f"Objects field is not a dictionary: {objects}")
                    return {}

                # Clean objects dictionary
                cleaned_objects = {}
                for obj_name, quantity in objects.items():
                    obj_name_clean = str(obj_name).lower().strip()
                    try:
                        quantity_int = int(quantity)
                        if obj_name_clean and quantity_int > 0:
                            cleaned_objects[obj_name_clean] = quantity_int
                    except (ValueError, TypeError):
                        log.warning(
                            f"Could not convert quantity '{quantity}' to integer for object '{obj_name_clean}'"
                        )

                # Clean numbers_found list
                cleaned_numbers = []
                if isinstance(numbers_found, list):
                    cleaned_numbers = [
                        str(num).strip() for num in numbers_found if str(num).strip()
                    ]

                result = {"objects": cleaned_objects, "numbers_found": cleaned_numbers}

                log.debug(f"LLM extracted numeracy relationship: {result}")
                return result

            else:
                log.warning(f"Could not extract JSON from LLM response: {response}")
                log.info("Falling back to regex-based numeracy extraction")
                return self._fallback_numeracy_extraction(prompt, available_numbers)

        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse JSON from LLM response: {e}")
            log.debug(f"Raw LLM response: {response}")
            log.info("Falling back to regex-based numeracy extraction")
            return self._fallback_numeracy_extraction(prompt, available_numbers)
        except Exception as e:
            log.error(f"Error extracting numeracy relationship with LLM: {e}")
            log.info("Falling back to regex-based numeracy extraction")
            return self._fallback_numeracy_extraction(prompt, available_numbers)

    def extract_spatial_relationship(
        self, prompt: str, available_relationships: List[str]
    ) -> Dict[str, str]:
        """
        Extract spatial relationship with two objects from the prompt using LLM.

        Args:
            prompt: Text prompt containing spatial relationship
            available_relationships: List of valid spatial relationships (from SPATIAL_QUERIES)

        Returns:
            Dictionary with 'obj1', 'relationship', and 'obj2' keys, or empty dict if extraction fails
        """
        llm_prompt = f"""Extract the spatial relationship and objects from the following text prompt.

Text Prompt: "{prompt}"

Available spatial relationships: {available_relationships}

Identify:
1. The first object (obj1) - the reference object
2. The spatial relationship - must be exactly one from the available relationships
3. The second object (obj2) - the object being positioned relative to obj1

Return the result as a JSON object with exactly these keys:
- "obj1": the first/reference object (lowercase, singular form)
- "relationship": the spatial relationship (exactly as written in available relationships)
- "obj2": the second object (lowercase, singular form)

For example:
- "a red car on the left of a blue bike" would return: {{"obj1": "car", "relationship": "on the left of", "obj2": "bike"}}
- "dog next to the table" would return: {{"obj1": "dog", "relationship": "next to", "obj2": "table"}}
- "cat on top of the chair" would return: {{"obj1": "cat", "relationship": "on top of", "obj2": "chair"}}
- "apple on the right of orange" would return: {{"obj1": "apple", "relationship": "on the right of", "obj2": "orange"}}

Important:
- Use the relationship exactly as it appears in the available relationships list
- Convert object names to lowercase, singular form
- If no valid spatial relationship is found, return an empty object: {{}}
- Focus on the main spatial relationship in the prompt

CRITICAL: Respond with ONLY valid JSON syntax. Do not include any explanations, reasoning, or additional text outside the JSON object."""

        try:
            response = self.llm.call(llm_prompt)

            # More robust JSON extraction that handles various response formats
            json_str = self._extract_json_from_response(response)
            if json_str:
                spatial_data = json.loads(json_str)

                # Validate the response structure
                if not isinstance(spatial_data, dict):
                    log.warning(f"LLM response is not a dictionary: {spatial_data}")
                    return {}

                # Check for empty response (valid case when no spatial relationship found)
                if not spatial_data:
                    log.debug("LLM found no spatial relationship in prompt")
                    return {}

                # Validate required keys
                required_keys = ["obj1", "relationship", "obj2"]
                if not all(key in spatial_data for key in required_keys):
                    log.warning(
                        f"Missing required keys in LLM response. Got: {spatial_data.keys()}, Expected: {required_keys}"
                    )
                    return {}

                # Clean and validate the extracted data
                obj1 = str(spatial_data["obj1"]).lower().strip()
                relationship = str(spatial_data["relationship"]).strip()
                obj2 = str(spatial_data["obj2"]).lower().strip()

                # Validate that relationship is in available relationships
                if relationship not in available_relationships:
                    log.warning(
                        f"Extracted relationship '{relationship}' not in available relationships: {available_relationships}"
                    )
                    return {}

                # Ensure objects are not empty
                if not obj1 or not obj2:
                    log.warning(
                        f"Empty object names extracted: obj1='{obj1}', obj2='{obj2}'"
                    )
                    return {}

                result = {"obj1": obj1, "relationship": relationship, "obj2": obj2}

                log.debug(f"LLM extracted spatial relationship: {result}")
                return result

            else:
                log.warning(f"Could not extract JSON from LLM response: {response}")
                return {}

        except json.JSONDecodeError as e:
            log.error(f"Failed to parse JSON from LLM response: {e}")
            log.debug(f"Raw LLM response: {response}")
            return {}
        except Exception as e:
            log.error(f"Error extracting spatial relationship with LLM: {e}")
            return {}


# Backward compatibility function
def extract_object_attributes_llm(llm, prompt: str) -> Dict[str, List[str]]:
    """
    Extract all objects and their attributes from the prompt using LLM.

    This function is provided for backward compatibility.
    Consider using ObjectAttributeExtractor class for better performance.

    Args:
        llm: LLM instance to use for extraction
        prompt: Text prompt

    Returns:
        Dict mapping object names to their list of attributes
    """
    extractor = ObjectAttributeExtractor(llm)
    return extractor.extract(prompt)


def criterion_index(idx):
    return f"crit_{idx:03}"


def yes_or_no(line):
    line = line.strip().replace("\n", " ")
    line = re.sub(r"[^a-zA-Z\s]", "", line)
    words = line.lower().split(" ")
    has_yes = "yes" in words
    has_no = "no" in words
    # log.debug(f"Line {line} words {words} has_yes={has_yes} has_no={has_no}")
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    if has_yes and has_no:
        return "invalid"
    return None


def list_from_text(text):
    """Try and parse text into a list"""

    def fmt(d):
        return "\\n" if d == "\n" else d

    text = text.strip("[] ")
    for delimiter in ["\n", ",", " "]:
        if delimiter in text:
            log.warning(f"Splitting text on delimiter: '{fmt(delimiter)}'")
            res = [yes_or_no(word.strip(" '\"")) for word in text.split(delimiter)]
            return [word for word in res if word]

    # No delimiters
    return [yes_or_no(text.strip(" '\""))]
