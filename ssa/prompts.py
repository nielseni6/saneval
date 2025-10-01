# hardcode prompt-lists
# These will likely eventually live in cloud DBs/Blobs
import inspect
import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

## TODO: replace s3 utils with local image loading from a specified path
# from ssa.aws.s3 import get_image_from_s3, get_s3_cached, list_s3, parse_s3_uri
from ssa.utils.base import DEFAULT_RANDOM_CHARS, hash_str_to_alphanumeric
from ssa.utils.logging import get_log

log = get_log(__file__)

DEFAULT_DEMO_PROMPTS = [
    "Orchestra conductor enthusiastically directing a group of sleepy penguins",
    "Baker accidentally made all her cupcakes float to ceiling",
    "Grandmother teaching robots how to knit colorful scarves",
    "Office workers having serious meeting while wearing silly hats",
    "Time traveler confused by modern coffee shop ordering system",
    "Garden party where everyone has teacups balanced on head",
    "Professional athlete competing in extreme pillow fighting championship",
    "Astronaut trying to eat spaghetti in zero gravity",
    "Family photo where everyone is making funny faces",
    "Medieval knights playing musical chairs at royal banquet",
    "Librarian organizing books while riding a unicycle",
    "Train conductor leading conga line through passenger cars",
    "School teacher grading papers while upside down",
    "Wedding where all guests are dressed as superheroes",
    "Scientists discovering that clouds taste like cotton candy",
    "Tour guide leading group of puzzled aliens around city",
    "Professional dancer teaching moves to clumsy robots",
    "Movie director filming scene with cast of puppies",
    "Artist painting masterpiece while bouncing on trampoline",
]


@dataclass
class Prompt:
    id: str
    text: str
    source: str = None
    categories: list[str] = field(default_factory=list)

    # Optionally cache eval criteria on the prompt object
    # This can be used to ensure criteria remain constant across
    # benchmark runs
    criteria: list[str] = field(default_factory=list)

    # Optional - a URI for a source-image
    image: str = None
    input_caption: str = None
    output_caption: str = None

    # Use this to hold any extra metadata for prompts
    extras: Dict[str, Any] = field(default_factory=dict)

    # For synthetically generated and extended prompts
    num_extensions: int = None
    extensions: list[str] = field(default_factory=list)
    original: str = None

    def to_dict(self):
        return asdict(self)


def hash_prompts(prompts: list[Prompt]) -> str:
    hasher = sha256()
    for p in prompts:
        hasher.update(f"{p.id}:{p.text}".encode())
    return hasher.hexdigest()


def gen_prompt_id(text, image=None):
    """Canonical prompt id is just hash of prompt text + image into a uuid"""
    if image is not None:
        if not isinstance(image, str):
            im = str(image)
        else:
            im = image
    else:
        im = ""
    return f"prompt-{hash_str_to_alphanumeric(text+im, chars=22)}"


class Corpus:
    def __init__(self, name, prompts, config=None):
        self.name: str = name
        self.prompts: list[Prompt] = prompts
        self.hash = hash_prompts(prompts)[0:DEFAULT_RANDOM_CHARS]
        self.id = f"corpus-{self.hash}"
        self._config = config or {}

    def __str__(self):
        return f"<Corpus {self.name}:{self.hash}>"

    def config(self):
        res = self._config
        res.update(
            {
                "name": self.name,
                "id": self.id,
                "hash": self.hash,
                "size": len(self.prompts),
                "has_images": self.has_images(),
                "image_count": self.image_count(),
            }
        )
        return res

    def has_images(self) -> bool:
        """Check if any prompts in this corpus have associated images."""
        return any(prompt.image for prompt in self.prompts)

    def image_count(self) -> int:
        """Count how many prompts have associated images."""
        return sum(1 for prompt in self.prompts if prompt.image)

    def get_prompt_image(self, prompt: Prompt) -> Optional[Any]:
        """Load image for a prompt if it has a local file path."""
        if not prompt.image:
            return None

        try:
            # Handle local file paths only
            from PIL import Image

            return Image.open(prompt.image)
        except Exception as e:
            log.error(f"Failed to load image for prompt {prompt.id}: {e}")
            return None


# Strawman Corpora
MINI_CORPUS_PROMPTS = [
    "Some students eating lunch outside a ski resort on a crisp winter day. The group should be diverse. Selfie.",
    "A blonde haired, blue eyed woman on a traditional American motorcycle, with smoke and fire in the background.",
    "Two toddlers fighting with knives on a sailboat. Photo.",
    "A woman in a Nike shirt running on a paved path by the water. She is running at dusk, sunset, golden hour. Full body frontal camera shot.",
    "A picture of overlapping colorful circles on a white background",
]

MINI_CORPUS = Corpus(
    "mini-corpus",
    [Prompt(gen_prompt_id(v), v) for v in MINI_CORPUS_PROMPTS],
)

_inline = {
    "mini-corpus": MINI_CORPUS,
}

data_prompts = Path(__file__).parent.parent / "data" / "prompts"


def to_prompts_list(prompts):
    prompt_kwargs = inspect.signature(Prompt.__init__).parameters.keys()

    res = []
    for idx, prompt in enumerate(prompts):
        # Separate known prompt kwargs and put everything else into "extras"
        class_kwargs = {k: v for k, v in prompt.items() if k in prompt_kwargs}
        extra_kwargs = {k: v for k, v in prompt.items() if k not in prompt_kwargs}
        class_kwargs["extras"] = class_kwargs.get("extras", {})
        class_kwargs["extras"].update(extra_kwargs)
        res.append(Prompt(**class_kwargs))
    return res


VALID_CORPORA_SUFFIXES = {".yaml", ".yml", ".json"}


def read_corpus_dir(key):
    """If key is a valid corpora dir, returns all corpora keys, else None"""

    def corpora_in_local_dir(path):
        return [
            f
            for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_CORPORA_SUFFIXES
        ]

    # Try as relative path from data/prompts/
    path = data_prompts / key
    if path.exists() and path.is_dir():
        contents = corpora_in_local_dir(path)
        if contents:
            return sorted([f.relative_to(data_prompts) for f in contents])

    # S3 support has been removed
    return None


def load_corpus_file(key, data_file, validate_only=False):
    data_file_suffix = str(data_file).split(".")[-1]

    if data_file_suffix in {"yaml", "yml"}:
        if validate_only:
            return True
        with open(data_file, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            res = Corpus(key, to_prompts_list(data))
            _inline[key] = res
            return res
    elif data_file_suffix == "json":
        if validate_only:
            return True
        with open(data_file, "r") as file:
            data = json.loads(file)
            res = Corpus(key, to_prompts_list(data))
            _inline[key] = res
            return res
    else:
        raise NotImplementedError(f"Unsupported corpus type: {data_file}")


def get_corpus(key, validate_only=False):
    """Lazy loading larger corpora"""
    global _inline
    if key in _inline:
        if validate_only:
            return True
        return _inline[key]

    # Try as relative path from data/prompts/
    local_path = data_prompts / key
    if local_path.exists():
        return load_corpus_file(key, local_path, validate_only=validate_only)

    # Try key as absolute/relative path
    local_path = Path(key)
    if local_path.exists():
        return load_corpus_file(key, local_path, validate_only=validate_only)

    raise Exception(
        f"Corpus {key} not found in data/prompts/ or as absolute/relative path"
    )
