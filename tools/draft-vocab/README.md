# llama-draft-vocab

Builds the token list for `--spec-draft-vocab`: the tokens an MTP draft head projects onto.

An MTP draft step multiplies its hidden state with the whole output head (for a 248k vocabulary that is about 1 GB
of weights per drafted token) and samples over the whole vocabulary on the CPU. Most of those tokens almost never
come up in what a model writes. With `--spec-draft-vocab FILE` the model gets a copy of the head's rows for the tokens
in the file at load (same type, split like the head under `-sm tensor`), and a draft step reads and samples only those.
The target still checks every drafted token against the whole vocabulary, so the output does not change; a token
outside the list is just never drafted.

```sh
llama-draft-vocab -m model.gguf -f code.txt -f prose.txt --coverage 0.998 -o draft-vocab.txt
llama-server -m model.gguf --spec-type draft-mtp --spec-draft-vocab draft-vocab.txt ...
```

- `-f FILE` (several allowed): text like the one the model will write: code in the languages you use, the
  languages of the answers, the format of the tool calls. The list is only as good as the corpus: in a language the
  corpus lacks, the drafts from the list are mostly rejected (German with a code/English list: a quarter accepted
  instead of half, slower than no list). While more than `--spec-draft-vocab-max-miss` (default 0.10) of the last 64
  tokens of a sequence are outside the list, the drafter uses the whole vocabulary instead, so such a conversation
  runs as without a list; include the languages you use in the corpus to keep the gain there too.
- `--coverage P` (default 0.995): the most frequent tokens that together make up this share of the corpus's tokens.
- `--max-tokens N`: at most N tokens.
- Control, user-defined (e.g. `<think>`, `<tool_call>`) and end-of-generation tokens are always added.

The output has one token id per line (`#` starts a comment); `--spec-draft-vocab` reads the same format, so a list can
also be written by hand or by another tool. The ids depend on the tokenizer, not on the weights: a list works for
every model with the same vocabulary.
