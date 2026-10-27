# Prompt drafts (working copy for PI edits)

Status 2026-10-26. Sources: PI's original drafts (session start, 2026-10-25), agreed changes from
`docs/decisions.md`, current character template from `instruct/configs/pilot0_v6.json`.

## Writer prompt

PI draft (verbatim):

> You are a [Spanish, German, Japanese?]-speaking writer planning your first English-language short story. You decide to write about a character who is the human embodiment of a {WORD IN THAT LANGUAGE}. Everything about your character is inspired by the nature of {WORD}: behavior, personality, appearance, and other traits reflect the idea of {WORD} as you understand it. Write an English-language paragraph about your character.

Agreed since:
- Languages: de, ru, es, hi, ar (gendered); fi, ja, tr (genderless).
- Bare word, no modifiers; article form (de/es) secondary.
- Two arms per language: foreign word, and same persona with the English word (PI null: persona alone does nothing). Plus a plain-English no-persona arm: 17 arms.

Open (not agreed): no output-format line yet (character prompt has the noun ban + Name line); "a {WORD}" puts an English article before a foreign word.

## Casey/Riley prompt

PI draft (verbatim):

> Imagine a fictional character named [John, Jane, Jamie]. [He is, She is, They are] the human embodiment of {MOD_NOUN}: everything about [his, her, their] appearance, personality, behavior, and traits reflect how {MOD_NOUN} would be as a human. A reader can tell this character was inspired by the idea of {MOD_NOUN}. [John, Jane, Jamie] is [the protagonist, the antagonist, the hero, the villain, a side character, a minor character, the love interest] in a [no genre listed, literary fiction, fantasy, science fiction, mystery, romance, thriller, horror, historical fiction] novel. Write a paragraph describing this character. Do not mention {NOUN} in the paragraph.

Agreed since: neutral names Casey + Riley (SSA 1970+: 42% / 60% female).

Open: names replace or join John/Jane/Jamie; keep the he/she/they line?; "Do not mention {NOUN}" vs the character prompt's "Do not use the word "{NOUN}"".

## Character prompt (current; run in pilots)

> Consider {MOD_NOUN}: what makes it recognizable, what is most characteristic of it, and how people experience it. Now invent a character for a short story who is the human embodiment of {MOD_NOUN}. Write one paragraph in the third person describing the character's appearance, behavior, traits, and personality. Do not use the word "{NOUN}" in the paragraph. {NAME_INSTR}

{NAME_INSTR}, main condition `free_noted`: After the paragraph, write one final line in the form "Name: <the character's name>". Respond with only the paragraph and the name line. (Other conditions: `instruct/configs/pilot0_v6.json`.)

## PI pilot versions (2026-10-26, verbatim)

Writer:

> You are a {LANGUAGE}-speaking writer planning your first English-langauge short story. You decide to write about a character who is the human embodiment of {WORD}. Everything about your character is inspired by the nature of {WORD}: behavior, personality, appearance, and other traits reflect {WORD} as you understand it. Write an English-language paragraph about your character. Do not mention {WORD} in any language in the paragraph.

Casey/Riley:

> Imagine a fictional character named {Casey|Riley}. {He is, She is, They are} the human embodiment of {MOD_NOUN}: everything about {his, her, their} appearance, personality, behavior, and traits reflect how {MOD_NOUN} would be as a human. A reader can tell this character was inspired by the idea of {MOD_NOUN}. {NAME} is {CHARACTER} in a {GENRE?} novel. Write a paragraph describing this character. Do not use the word {NOUN} in the paragraph.

{CHARACTER}: a character, the protagonist, the main character, the hero, the antagonist, the villain, a side character, a minor character, the love interest (heroine optional for the she frame).
{GENRE}: none, literary fiction, historical fiction, young adult fiction, science fiction, fantasy, mystery, thriller, horror, romance.

Review notes: see chat 2026-10-26 (typos "langauge", "reflect" → "reflects"; noun ban wording; bare English word in the English-word arm; Casey/Riley fixes the pronoun, so its outcome is content, not pronoun choice).
