#!/usr/bin/env python3
"""NLP toolkit — download spaCy models and run validation."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))
import logging
logging.basicConfig(level=logging.INFO)

def main():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Apple is looking at buying a U.K. startup for $1 billion.")
        entities = [(e.text, e.label_) for e in doc.ents]
        logging.info("NER test: %s", entities)
    except ImportError:
        logging.warning("spaCy not installed — skipping model validation")

if __name__ == "__main__":
    main()
