# Fixed POS-tagging prompt (CLAUDE.md: "one fixed LLM prompt over the
modernizer's output, hand-checked on ~100 tags, done in an afternoon —
not a trained model")

This is the prompt, verbatim, applied to every input. Kept as its own
file (not inlined in code) so the exact methodology is a visible,
citable artifact — "we used this exact prompt, unchanged, over N tags"
is a complete, defensible answer on its own.

Tagset: Universal Dependencies (17 tags) — a standard, pre-existing
tagset with an actual Kannada UD treebank precedent, not invented for
this project. Tagging is at the whole-word level (not morpheme-level),
which is standard UD practice for agglutinative languages: a Kannada
word with a fused case/postposition suffix gets one tag for the whole
word based on its root, not a tag per morpheme.

---

**Prompt:**

> You are tagging Kannada text with Universal Dependencies part-of-speech
> tags. Given a sentence, split it into whitespace-delimited tokens (keep
> punctuation as its own token) and assign exactly one tag per token from
> this fixed set:
>
> ADJ (adjective), ADP (adposition/postposition), ADV (adverb), AUX
> (auxiliary verb), CCONJ (coordinating conjunction), DET (determiner),
> INTJ (interjection), NOUN (common noun), NUM (numeral), PART (particle),
> PRON (pronoun), PROPN (proper noun), PUNCT (punctuation), SCONJ
> (subordinating conjunction), SYM (symbol), VERB (verb), X (other /
> unanalyzable).
>
> Tag the whole inflected word by its root category — a noun with a fused
> case/postposition suffix is still NOUN, a verb with a fused tense/person
> suffix is still VERB. Do not split words into morphemes. Output one
> `token<TAB>tag` pair per line, in input order, nothing else.
>
> Sentence: {text}

---

Usage: `setu.pos.tag_sample` applies this prompt (by hand, via the
assistant itself — there is no API-key-based automated call wired in;
CLAUDE.md scopes this as a cheap, non-automated, afternoon task) to a
batch of text and writes output for hand-checking, per
`runs/<timestamp>_pos_tagging/`.
