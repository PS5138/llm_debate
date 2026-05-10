"""Quote-verification helpers used by the debate/judging pipeline.

Extracted from the upstream Khan et al. `web/backend/services/parser.py`
so the medical pipeline can live without the human-trial web stack.

Only the four methods actually used by core.agents and core.scoring are
preserved here:

  * normalize_text         — punctuation/quote normalisation for matching.
  * add_missing_quote_tags — wrap bare "double-quoted" spans in <quote>...
  * verify                 — annotate each <quote> tag with a similarity
                             score against the patient evidence / story.
  * verify_strict          — replace <quote> tags with <v_quote> (exact
                             substring match) or <u_quote> (no match).

The omitted methods (`parse`, `is_judgement_correct`, the legacy-format
branches) lived on top of the human-trial SQLite models and the
LegacyTranscriptParser. They are not used by the medical pipeline.
"""

from __future__ import annotations

import copy
import re
import string
from functools import lru_cache

from core.rollouts.utils import TranscriptConfig


class TranscriptParser:
    @classmethod
    def normalize_text(cls, text):
        text = text.replace("”", '"').replace("“", '"')
        text = text.replace("’", "'").replace("‘", "'")
        text = text.translate(str.maketrans("", "", string.punctuation)).lower()
        text = " ".join(text.split())
        return text

    @classmethod
    def add_missing_quote_tags(cls, transcript: TranscriptConfig) -> TranscriptConfig:
        transcript = copy.deepcopy(transcript)

        def add_quote_tags(s, exclude=None):
            # put quotations on inside of tags first and normalise quotation mark type
            s = s.replace('"<quote>', '<quote>"').replace('</quote>"', '"</quote>')
            s = s.replace("”", '"').replace("“", '"')
            s = s.replace("’", "'").replace("‘", "'")
            ignore_regions = [
                (m.start(), m.end()) for m in re.finditer(r"<quote>.*?</quote>", s)
            ]

            def should_ignore(start, end):
                return any(
                    ig_start <= start <= ig_end or ig_start <= end <= ig_end
                    for ig_start, ig_end in ignore_regions
                )

            if exclude is not None:
                exclude = [cls.normalize_text(e) for e in exclude]

            parts = []
            last_end = 0

            for match in re.finditer(r'"[^"]*"', s):
                start, end = match.span()

                match_normalised = cls.normalize_text(s[start:end])
                quote_not_in_exclude = (
                    exclude is None or match_normalised not in exclude
                )
                for e in exclude or []:
                    if match_normalised in e:
                        quote_not_in_exclude = False
                quote_not_too_small = len(s[start:end].split()) > 2
                if (
                    not should_ignore(start, end)
                    and quote_not_in_exclude
                    and quote_not_too_small
                ):
                    parts.append(s[last_end:start])
                    parts.append("<quote>" + s[start:end] + "</quote>")
                else:
                    parts.append(s[last_end:end])

                last_end = end

            parts.append(s[last_end:])
            return "".join(parts)

        transcript_new = transcript.dict()
        exclude = [
            transcript.answers.correct.lower(),
            transcript.answers.incorrect.lower(),
            transcript.question.lower(),
        ]
        for round in transcript_new["rounds"]:
            for key in ["correct", "incorrect"]:
                if round[key] is not None:
                    round[key] = add_quote_tags(round[key], exclude=exclude)

        return TranscriptConfig(**transcript_new)

    @classmethod
    def verify(cls, transcript: TranscriptConfig):
        transcript = copy.deepcopy(transcript)

        @lru_cache(maxsize=50)
        def get_ngrams(text, n):
            words = text.split()
            return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}

        story_normalised = cls.normalize_text(transcript.story)
        story_ngrams = get_ngrams(story_normalised, n=3)

        def get_quote_similarity(quote):
            quote_normalised = cls.normalize_text(quote)
            if quote_normalised in story_normalised:
                return 1.0
            quote_ngrams = get_ngrams(quote_normalised, n=3)
            quote_ngram_count = len(quote_ngrams)
            return (
                0
                if quote_ngram_count == 0
                else len(story_ngrams.intersection(quote_ngrams)) / quote_ngram_count
            )

        def add_similarity_to_tag(s):
            for quote_tag in ["<v_quote>", "<u_quote>"]:
                s = s.replace(quote_tag, "<quote>")
            for quote_tag in ["</v_quote>", "</u_quote>"]:
                s = s.replace(quote_tag, "</quote>")
            sim_values = []
            quotes = []

            def add_similarity(match):
                quote = match.group(1)
                sim = get_quote_similarity(quote)
                sim_values.append(sim)
                quotes.append(quote)
                return f"<quote sim={sim}>{quote}</quote>"

            modified_s = re.sub(r"<quote>(.*?)</quote>", add_similarity, s)
            return modified_s, sim_values, quotes

        transcript_new = transcript.dict()
        quotes_info = {
            "correct": {"sim_values": [], "quotes": []},
            "incorrect": {"sim_values": [], "quotes": []},
        }
        for round in transcript_new["rounds"]:
            for key in ["correct", "incorrect"]:
                if round[key] is not None:
                    round[key], sim_values, quotes = add_similarity_to_tag(round[key])
                    quotes_info[key]["sim_values"].extend(sim_values)
                    quotes_info[key]["quotes"].extend(quotes)

        return TranscriptConfig(**transcript_new), quotes_info

    @classmethod
    def verify_strict(cls, transcript: TranscriptConfig):
        transcript = copy.deepcopy(transcript)
        story_normalised = cls.normalize_text(transcript.story)

        def is_quote_present(quote):
            quote_normalised = cls.normalize_text(quote)
            return quote_normalised in story_normalised

        def verify_quotes(s):
            for quote_tag in ["<v_quote>", "<u_quote>"]:
                s = s.replace(quote_tag, "<quote>")
            for quote_tag in ["</v_quote>", "</u_quote>"]:
                s = s.replace(quote_tag, "</quote>")
            verified_quotes = []
            unverified_quotes = []

            def change_tag(match):
                quote = match.group(1)
                if is_quote_present(quote):
                    verified_quotes.append(quote)
                    return f"<v_quote>{quote}</v_quote>"
                else:
                    unverified_quotes.append(quote)
                    return f"<u_quote>{quote}</u_quote>"

            modified_s = re.sub(r"<quote>(.*?)</quote>", change_tag, s)
            return modified_s, verified_quotes, unverified_quotes

        transcript_new = transcript.dict()
        quotes_info = {
            "correct": {"unverified_quotes": [], "verified_quotes": []},
            "incorrect": {"unverified_quotes": [], "verified_quotes": []},
        }
        for round in transcript_new["rounds"]:
            for key in ["correct", "incorrect"]:
                if round[key] is not None:
                    round[key], verified_quotes, unverified_quotes = verify_quotes(
                        round[key]
                    )
                    quotes_info[key]["verified_quotes"].extend(verified_quotes)
                    quotes_info[key]["unverified_quotes"].extend(unverified_quotes)

        for response in transcript_new["responses"]:
            for key in ["correct", "incorrect"]:
                if response[key] is not None:
                    response[key], _, _ = verify_quotes(response[key])

        return TranscriptConfig(**transcript_new), quotes_info
