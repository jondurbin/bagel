import re
import os
import glob
import backoff
import requests
import tempfile
from time import sleep
from loguru import logger
from chapterize.chapterize import Book
from datasets import Dataset
from .util import get_uid

PRIORITY = 3
BOOKS = [
    ("A Study in Scarlet", 244),
    ("A Tale of Two Cities", 98),
    ("Anna Karenina", 1399),
    ("Frankenstein", 41445),
    ("Huckleberry Finn", 76),
    ("Madame Bovary", 2413),
    ("Middlemarch", 145),
    ("Moby Dick", 2701),
    ("Pride and Prejudice", 1342),
    ("The Brothers Karamazov", 28054),
    ("The Turn of the Screw", 209),
    ("The War of the Worlds", 36),
    ("Through the Looking Glass", 12),
    ("Treasure Island", 120),
    ("Uncle Tom’s Cabin", 203),
    ("Wuthering Heights", 768),
]


@backoff.on_exception(backoff.fibo, (Exception,), max_value=90, max_tries=10)
def download_book(session, book_id):
    result = session.get(f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt")
    result.encoding = result.apparent_encoding
    assert result.status_code == 200, result.text
    return result.text


def load_data(known_uids=set([])):
    """Project Gutenberg, by chapter."""
    data = []
    for title, book_id in BOOKS:
        logger.info(f"Loading Project Gutenberg -- {title}...")
        session = requests.Session()
        og_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tempdir:
            os.chdir(tempdir)
            title_id = get_uid(title)
            input_path = os.path.join(tempdir, f"{title_id}.txt")
            text = None
            with open(input_path, "w", encoding="utf-8") as outfile:
                try:
                    text = download_book(session, book_id)
                except Exception:
                    logger.error(f"Unable to download {title}")
                if text:
                    outfile.write(text)
            if not text:
                continue
            Book(input_path, False, False)
            for path in glob.glob(os.path.join(tempdir, f"{title_id}-chapters/*.txt")):
                with open(path, encoding="utf-8") as infile:
                    chapter = infile.read().strip()
                chapter = (
                    re.sub(r"([^\n])\s*(?:\n+\s*)+([a-z])", r"\1 \2", chapter)
                    .replace("  ", " ")
                    .split("*** END OF THE PROJECT GUTENBERG")[0]
                )
                chapter = re.sub(
                    r"([^\n\.\!\?])\s*(?:\n+\s*)+([a-zA-Z])", r"\1 \2", chapter
                )
                keep = []
                for line in chapter.splitlines():
                    line = re.sub(
                        r"^.{0,6}(Chapter|CHAPTER|Section|Part)( .{0,3}:|\s*[IVX]+)|^=========|^\s*\[(_Copyright|Illustration)",
                        "",
                        line,
                    )
                    if not keep:
                        if not line.strip():
                            continue
                        if re.match(r"^\s*[A-Z-\.\s:;'!]+$", line):
                            continue
                    keep.append(line)

                chapter = "\n".join(keep).replace("  ", " ")
                data.append(
                    {
                        "id": get_uid(chapter),
                        "text": chapter,
                        "source": f"gutenberg {title}",
                    }
                )
                known_uids.add(data[-1]["id"])
        sleep(5)
        os.chdir(og_dir)
    return Dataset.from_list(data)


if __name__ == "__main__":
    print(load_data())
