"""Starter conversations that turn a FeatherLM base model into a small chatbot.

All text here was written for this project (MIT licensed). Shakespeare quotes are taken
from the public-domain training corpus when available.

The bot is deliberately honest about its size: for questions outside what it was taught,
it answers that it does not know yet and points to the Teach tab.

    python -m featherlm.data.chat_starter --out data/chat/starter --corpus data/shakespeare/raw/tinyshakespeare.txt

Developed by Ammad Aslam with AI-assisted tooling.
"""
from __future__ import annotations

import argparse
import json
import random
import zlib
from pathlib import Path

REPO_URL = "https://github.com/engr-dr-ammad/FeatherLM"

# intent -> (question phrasings, answer variants). The last 2 phrasings of each intent are held out for evaluation.
INTENTS: dict[str, tuple[list[str], list[str]]] = {
    "greet": ([
        "hi", "hello", "hey", "hello there", "hi there", "hey there", "good morning", "good evening",
        "good afternoon", "hey featherlm", "hello featherlm", "hiya", "greetings", "yo", "howdy", "hi featherlm",
    ], [
        "Hello! I'm FeatherLM, a small language model. How can I help you today?",
        "Hi there! I'm FeatherLM. What would you like to know?",
    ]),
    "how_are_you": ([
        "how are you", "how are you doing", "how's it going", "how do you feel", "are you okay", "how are things",
        "what's up", "how have you been",
    ], [
        "I'm doing well, thanks for asking! How can I help you?",
        "All good here, thank you! What can I do for you?",
    ]),
    "identity": ([
        "who are you", "what are you", "what's your name", "what is your name", "introduce yourself",
        "tell me about yourself", "are you a robot", "are you an ai", "what kind of ai are you", "describe yourself",
        "who am i talking to", "what should i call you",
    ], [
        "I'm FeatherLM, a small open-source language model with about 13 million parameters. I was developed by "
        "Ammad Aslam, and I can be trained and run on an ordinary CPU.",
    ]),
    "creator": ([
        "who made you", "who created you", "who built you", "who developed you", "who is your creator",
        "who wrote featherlm", "who is the author of featherlm", "who designed you", "who programmed you",
        "who is behind featherlm",
    ], [
        "I was developed by Ammad Aslam, with AI-assisted tooling.",
    ]),
    "what_is": ([
        "what is featherlm", "explain featherlm", "what does featherlm do", "tell me about featherlm",
        "what is this project", "what is this model", "what makes featherlm special", "how does featherlm work",
        "what is special about you", "what is featherlm used for",
    ], [
        "FeatherLM is a hardware-efficient language model. It can be trained from scratch on a normal CPU, and "
        "while it writes, its memory stays at about 0.15 MB no matter how long the text gets.",
    ]),
    "benefits": ([
        "why should i use featherlm", "what are the benefits", "what are your advantages", "why featherlm",
        "what are the benefits of featherlm", "what is good about featherlm", "why is featherlm useful",
        "what are your strengths", "give me reasons to use featherlm", "what are the pros",
    ], [
        "FeatherLM trains on a normal CPU with no GPU, runs fully offline so your data stays private, keeps its "
        "memory flat on long texts, and the 8-bit version is 4 times smaller with the same quality. It is free "
        "and open source.",
    ]),
    "gpu": ([
        "do i need a gpu", "can it run on a cpu", "can i train you on my laptop", "does featherlm need a gpu",
        "can i run you without a gpu", "will it work on my laptop", "what hardware do i need",
        "can i train it without a graphics card", "is a gpu required", "does it run on a normal computer",
    ], [
        "No GPU is needed. FeatherLM trains and runs on an ordinary CPU. The Nano model trained on 2 CPU cores in "
        "about 40 minutes, with a peak of about 2 GB of RAM.",
    ]),
    "install": ([
        "how do i install featherlm", "how do i download featherlm", "where can i get featherlm",
        "how do i get you", "how to install", "how to download", "where is the code", "is featherlm on github",
        "how do i set it up", "how do i get started", "where can i download you", "how do i install you",
    ], [
        f"Clone it from GitHub with git clone {REPO_URL}, then run ./scripts/setup.sh. On Windows, run "
        "scripts\\setup.ps1 instead.",
    ]),
    "train": ([
        "how do i train you", "how do i train featherlm", "how to train", "how do i train a model",
        "how do i train from scratch", "what command trains the model", "how can i train my own model",
        "how long does training take", "how do i start training", "how do i train it on my data",
    ], [
        "First prepare data with python -m featherlm.data.prepare, then run python train.py --config "
        "configs/nano_cpu.yaml. The Nano model trains in about 40 minutes on 2 CPU cores.",
    ]),
    "teach": ([
        "how can i teach you", "can i teach you new things", "how do i customize you", "can i add my own data",
        "how do i make you smarter", "can you learn", "how do i fine tune you", "can i train you as a chatbot",
        "how do i teach you a new answer", "can you learn from me",
    ], [
        "Yes! Open the Teach tab in FeatherLM Studio, add questions with the answers you want, and click Train. "
        "When training finishes, chat with your new model. You can also click Teach under any of my replies to "
        "save a better answer.",
    ]),
    "limits": ([
        "what can't you do", "what are your limitations", "are you like chatgpt", "are you smart",
        "can you do anything", "how good are you", "what are your weaknesses", "do you know everything",
        "are you as good as gpt", "can you replace chatgpt",
    ], [
        "I'm a small research model, so I only know what I've been taught: things about FeatherLM, simple maths "
        "and a bit of Shakespeare. I'm not a general assistant like the big chatbots, but you can teach me more "
        "in the Teach tab.",
    ]),
    "memory": ([
        "how much memory do you use", "how much ram do you need", "how much memory does featherlm need",
        "is featherlm memory efficient", "how big are you", "how large is the model", "how many parameters do you have",
        "what is your model size", "how much ram does training use", "how heavy are you",
    ], [
        "The Nano model has about 13 million parameters. Training peaked at about 2 GB of RAM, and while I write, "
        "my memory state stays at about 0.15 MB however long the text is. The 8-bit version is about 13 MB.",
    ]),
    "license": ([
        "is featherlm free", "what is the license", "can i use featherlm for free", "is it open source",
        "how much does featherlm cost", "what license does featherlm use", "can i use it commercially",
        "do i have to pay", "is the code public", "is featherlm free to use",
    ], [
        "Yes, FeatherLM is free and open source under the MIT license.",
    ]),
    "privacy": ([
        "do you need the internet", "does featherlm work offline", "is my data private", "do you send my data anywhere",
        "can i use you offline", "is featherlm private", "do you store my messages online", "does it need wifi",
        "is my data safe", "do you work without internet",
    ], [
        "I run fully offline on your own device, so your data stays with you. Nothing is sent to the internet.",
    ]),
    "thanks": ([
        "thanks", "thank you", "thank you so much", "thanks a lot", "cheers", "much appreciated", "great, thanks",
        "thanks for the help", "awesome thank you", "ty",
    ], [
        "You're welcome! Anything else I can help with?",
        "Happy to help!",
    ]),
    "bye": ([
        "bye", "goodbye", "see you", "see you later", "good night", "talk to you later", "bye bye", "i have to go",
        "catch you later", "farewell",
    ], [
        "Goodbye! Have a great day.",
        "See you later!",
    ]),
    "joke": ([
        "tell me a joke", "make me laugh", "say something funny", "do you know any jokes", "tell me something funny",
        "got a joke", "i need a joke", "joke please",
    ], [
        "Why did the language model go to school? To improve its word choice!",
        "I tried to train on a GPU, but I was told I'm too light for heavy lifting. I'm a feather after all!",
        "Why don't small models get lost? They always follow the next token.",
    ]),
    "unknown": ([
        "what is the weather today", "what is the capital of france", "who won the football match",
        "what is the stock price of apple", "what's the news today", "can you book a flight", "what time is it",
        "translate this into spanish", "write me a python program", "what is the meaning of life",
        "who is the president", "recommend a good restaurant", "how do i cook pasta", "what is quantum physics",
        "explain black holes", "what year is it",
    ], [
        "I'm a small model and I don't know that yet. You can teach me new answers in the Teach tab of "
        "FeatherLM Studio!",
    ]),
}

SHAKESPEARE_Q = ["give me a shakespeare quote", "quote shakespeare", "say something like shakespeare",
                 "tell me a line from shakespeare", "recite some shakespeare", "speak like shakespeare",
                 "share a shakespeare line", "i want a shakespeare quote"]
MATH_TEMPLATES = [("what is {a} plus {b}", "plus"), ("what's {a} + {b}", "plus"), ("add {a} and {b}", "plus"),
                  ("{a} plus {b}", "plus"), ("what is {a} minus {b}", "minus"), ("what's {a} - {b}", "minus"),
                  ("subtract {b} from {a}", "minus")]


# More phrasings, inserted before each intent's two held-out phrasings.
EXTRA_PHRASINGS = {
 "greet": [
  "hello!",
  "hey hey",
  "morning",
  "hi friend",
  "hello friend",
  "nice to meet you",
  "hi, anyone there",
  "heya"
 ],
 "how_are_you": [
  "how is your day",
  "how are you today",
  "are you doing well",
  "how's your day going",
  "you good",
  "how do you do"
 ],
 "identity": [
  "what is your name and what do you do",
  "who exactly are you",
  "what am i chatting with",
  "tell me who you are",
  "what are you called",
  "what's your identity",
  "are you a chatbot",
  "are you a person"
 ],
 "creator": [
  "who owns featherlm",
  "who invented you",
  "who is your developer",
  "who made featherlm",
  "who coded you",
  "who started featherlm",
  "which person made you"
 ],
 "what_is": [
  "what exactly is featherlm",
  "can you explain featherlm",
  "describe featherlm",
  "what is featherlm all about",
  "give me an overview of featherlm",
  "what is the idea behind featherlm"
 ],
 "benefits": [
  "why is featherlm better",
  "what is featherlm good at",
  "list the benefits",
  "what do i gain from featherlm",
  "why would i choose featherlm",
  "what advantages does featherlm have"
 ],
 "gpu": [
  "can i run it on an old laptop",
  "do i need a graphics card",
  "does it need cuda",
  "will it run on my pc",
  "can it train on cpu only",
  "do you need special hardware"
 ],
 "install": [
  "how can i install it",
  "how do i download it",
  "what is the github link",
  "where do i find featherlm",
  "how do i clone featherlm",
  "give me the download link",
  "installation steps please"
 ],
 "train": [
  "how do i train the model",
  "what are the training steps",
  "how does training work",
  "train featherlm how",
  "what do i run to train",
  "teach me to train featherlm"
 ],
 "teach": [
  "can i teach you",
  "how do i add new answers",
  "can i give you my own answers",
  "how do i change your answers",
  "can you learn new answers",
  "how can i improve your answers"
 ],
 "limits": [
  "are you a general assistant",
  "what don't you know",
  "are you clever",
  "can you answer anything",
  "are you better than chatgpt",
  "what are you bad at"
 ],
 "memory": [
  "how many parameters",
  "what is your size",
  "how much ram do i need",
  "how big is featherlm",
  "how much memory does it use",
  "how small are you"
 ],
 "license": [
  "is it free",
  "what license is featherlm",
  "is featherlm open source",
  "can i use it for free",
  "is it paid",
  "is it free software"
 ],
 "privacy": [
  "is it private",
  "does it need an internet connection",
  "can it work offline",
  "do you share my data",
  "does my data leave my computer",
  "is it safe to use"
 ],
 "thanks": [
  "thank you very much",
  "thanks so much",
  "many thanks",
  "thx",
  "nice, thanks",
  "perfect, thank you"
 ],
 "bye": [
  "goodbye for now",
  "bye for now",
  "see ya",
  "have a nice day",
  "i'm leaving",
  "later"
 ],
 "joke": [
  "tell me a funny joke",
  "do you have a joke",
  "can you tell jokes",
  "another joke",
  "be funny"
 ],
 "unknown": [
  "what is the capital of germany",
  "how tall is mount everest",
  "what's the temperature outside",
  "who wrote harry potter",
  "what is 2 to the power of 64",
  "tell me today's headlines",
  "how do i fix my car",
  "what is the population of china"
 ]
}
for _k, _v in EXTRA_PHRASINGS.items():
    _qs = INTENTS[_k][0]
    INTENTS[_k][0][:] = _qs[:-2] + [q for q in _v if q not in _qs] + _qs[-2:]


def _vary(q: str, rng: random.Random) -> str:
    q = q.strip()
    r = rng.random()
    if r < 0.30:
        q = q[0].upper() + q[1:]
    elif r < 0.40:
        q = q.upper() if len(q) < 12 else q.capitalize()
    pre = rng.random()
    if pre < 0.08:
        q = "FeatherLM, " + q
    elif pre < 0.14:
        q = "please, " + q if not q.lower().startswith(("hi", "hello", "hey")) else q
    is_question = q.lower().startswith(("who", "what", "how", "where", "why", "is", "are", "can", "do", "does",
                                        "will", "should"))
    end = rng.random()
    if is_question and end < 0.6:
        q += "?"
    elif not is_question and end < 0.3:
        q += rng.choice(["!", "."])
    return q


def shakespeare_lines(corpus: str | None, n: int, rng: random.Random) -> list[str]:
    if corpus and Path(corpus).exists():
        text = Path(corpus).read_text(encoding="utf-8")
        lines = [l.strip() for l in text.splitlines()
                 if 30 <= len(l.strip()) <= 90 and not l.strip().endswith(":") and l.strip()[0].isupper()]
    else:
        lines = ["All the world's a stage, and all the men and women merely players.",
                 "The course of true love never did run smooth.", "Brevity is the soul of wit."]
    return [rng.choice(lines) for _ in range(n)]


def build(corpus: str | None = None, seed: int = 0, per_intent: int = 160):
    rng = random.Random(seed)
    train, val = [], []

    def conv(q, a):
        return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}

    for name, (qs, answers) in INTENTS.items():
        tr_q, va_q = qs[:-2], qs[-2:]
        for _ in range(per_intent):
            train.append(conv(_vary(rng.choice(tr_q), rng), rng.choice(answers)))
        for q in va_q:
            val.append({**conv(_vary(q, random.Random(zlib.crc32(q.encode()))), answers[0]), "intent": name})
    quotes = shakespeare_lines(corpus, 220, rng)
    for i, line in enumerate(quotes):
        q = SHAKESPEARE_Q[:-2][i % 6]
        train.append(conv(_vary(q, rng), f"Here is a line from Shakespeare: \"{line}\""))
    for q in SHAKESPEARE_Q[-2:]:
        val.append({**conv(q + "?", "Here is a line from Shakespeare:"), "intent": "shakespeare"})
    held_math = {(7, 5, "plus"), (13, 4, "minus"), (9, 9, "plus"), (18, 6, "minus")}
    n = 0
    while n < 900:
        tmpl, op = rng.choice(MATH_TEMPLATES)
        a, b = rng.randint(0, 20), rng.randint(0, 20)
        if op == "minus" and b > a:
            a, b = b, a
        if (a, b, op) in held_math:
            continue
        res = a + b if op == "plus" else a - b
        train.append(conv(_vary(tmpl.format(a=a, b=b), rng), f"{a} {op} {b} is {res}."))
        n += 1
    for a, b, op in sorted(held_math):
        res = a + b if op == "plus" else a - b
        val.append({**conv(f"What is {a} {op} {b}?", f"{a} {op} {b} is {res}."), "intent": "math"})
    # multi-turn: greeting then a question
    names = [n for n in INTENTS if n not in ("greet", "thanks", "bye")]
    for _ in range(400):
        g_q, g_a = INTENTS["greet"]
        intent = rng.choice(names)
        q2s, a2s = INTENTS[intent]
        msgs = [{"role": "user", "content": _vary(rng.choice(g_q[:-2]), rng)},
                {"role": "assistant", "content": rng.choice(g_a)},
                {"role": "user", "content": _vary(rng.choice(q2s[:-2]), rng)},
                {"role": "assistant", "content": rng.choice(a2s)}]
        if rng.random() < 0.3:
            msgs += [{"role": "user", "content": _vary(rng.choice(INTENTS["thanks"][0][:-2]), rng)},
                     {"role": "assistant", "content": rng.choice(INTENTS["thanks"][1])}]
        train.append({"messages": msgs})
    rng.shuffle(train)
    return train, val


def write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the FeatherLM starter chat dataset")
    ap.add_argument("--out", default="data/chat/starter")
    ap.add_argument("--corpus", default="data/shakespeare/raw/tinyshakespeare.txt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train, val = build(a.corpus, a.seed)
    out = Path(a.out)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl([{k: v for k, v in r.items() if k != "intent"} for r in val], out / "val.jsonl")
    write_jsonl(val, out / "eval_heldout.jsonl")
    (out / "README.txt").write_text("FeatherLM starter chat data (MIT). Generated by featherlm.data.chat_starter.\n"
                                    "eval_heldout.jsonl uses phrasings that never appear in train.jsonl.\n")
    print(f"train {len(train)} conversations, held-out eval {len(val)}")


if __name__ == "__main__":
    main()
