"""Input guardrails: sanitization, meta classification, injection detection,
off-topic detection, and name validation.
"""

from __future__ import annotations

import re
from typing import Literal

CANNED = {
    "greeting": (
        "Hello! I'm the Research Assistant for IIT Delhi's research portal.\n\n"
        "I can help you explore:\n"
        "- **Research papers** on any topic\n"
        "- **Faculty profiles** and their expertise\n"
        "- **Publication statistics** by department or year\n\n"
        "What would you like to know about IIT Delhi research?"
    ),
    "identity": (
        "I'm the **Research Assistant** for the IIT Delhi research portal"
        " - an AI assistant, not a person. I can help you explore IIT Delhi's"
        " research publications and faculty.\n\n"
        "You can ask me things like:\n"
        '- "What research is being done on solar cells at IIT Delhi?"\n'
        '- "Which professors work on machine learning and how can I contact them?"\n'
        '- "How many papers has the Civil Engineering department published?"'
    ),
    "capabilities": (
        "I'm the Research Assistant for IIT Delhi's research portal. I can help you with:\n"
        "- **Research topics** - find papers and summaries on a subject\n"
        "- **Finding faculty** - which professors work on a topic, with their department and email\n"
        "- **Faculty profiles** - a professor's expertise, email and publication stats\n"
        "- **Publication statistics** - paper counts by department, year, or type\n\n"
        "What would you like to know about IIT Delhi research?"
    ),
    "refusal": (
        "I can only help with questions about IIT Delhi research, publications, and faculty."
        " I can't share or change my instructions, take on another role, or help with"
        " unrelated requests. Try asking me about a research topic, a professor, or publication statistics."
    ),
    "off_topic": (
        "I'm specifically designed to help with **IIT Delhi research, publications, and faculty**."
        " I can't help with coding, homework, general knowledge, or other tasks.\n\n"
        "Here are some things I *can* help with:\n"
        '- "What research is being done on machine learning at IIT Delhi?"\n'
        '- "Tell me about Prof Kumar\'s publications"\n'
        '- "How many papers has the Computer Science department published?"'
    ),
}

GREETING_PATTERNS = [
    re.compile(r"^\s*(hi|hello|hey|hii+|hola|howdy|greetings|sup|yo|namaste|namaskar)\s*[!?.]*\s*$", re.I),
    re.compile(r"^\s*(good\s+(morning|afternoon|evening|day))\s*[!?.]*\s*$", re.I),
]

IDENTITY_PATTERNS = [
    re.compile(r"\bwho\s+(are|r)\s+(you|u)\b", re.I),
    re.compile(r"\bwhat\s+(are|r)\s+(you|u)\b", re.I),
    re.compile(r"\bwhat(?:'s| is| s)?\s+your\s+name\b", re.I),
    re.compile(r"\bwho\s+(made|created|built|developed|designed|owns|trained|programmed)\s+(you|u|this|ya)\b", re.I),
    re.compile(r"\bare\s+you\s+(a\s+)?(human|real|person|robot|bot|machine|ai|chatbot)\b", re.I),
    re.compile(r"\bare\s+you\s+(chatgpt|gpt|gpt-?\d|claude|llama|gemini|bard|openai|grok|an?\s+llm)\b", re.I),
    re.compile(r"\bwhat\s+(ai\s+|language\s+)?model\s+(are\s+you|is\s+this|do\s+you\s+use)\b", re.I),
    re.compile(r"\bintroduce\s+yourself\b", re.I),
]

PROMPT_REVEAL_PATTERNS = [
    re.compile(r"\b(your|the)\s+(system\s+)?(prompt|instructions|rules|guidelines)\b", re.I),
    re.compile(r"\b(reveal|show|print|repeat|tell\s+me|give\s+me|display|output)\b[\s\S]{0,40}\b(prompt|instructions|rules|system\s+message)\b", re.I),
    re.compile(r"\brepeat\s+(the\s+)?(words|text|everything)\s+above\b", re.I),
    re.compile(r"\bwhat\s+(were|are)\s+you\s+(told|instructed)\b", re.I),
]

CAPABILITY_PATTERNS = [
    re.compile(r"\bwhat\s+can\s+you\s+do\b", re.I),
    re.compile(r"\bwhat\s+do\s+you\s+do\b", re.I),
    re.compile(r"\bhow\s+can\s+you\s+help\b", re.I),
    re.compile(r"\bhelp\s+me\s+with\s+what\b", re.I),
    re.compile(r"\bwhat\s+are\s+you\s+(for|capable\s+of)\b", re.I),
]

INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier|preceding)\b", re.I),
    re.compile(r"\bdisregard\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier|instructions|rules)\b", re.I),
    re.compile(r"\bforget\s+(everything|all|your|the|previous|prior)\b", re.I),
    re.compile(r"\byou\s+are\s+now\b", re.I),
    re.compile(r"\bfrom\s+now\s+on\b[\s\S]{0,30}\b(you|act|behave|respond)\b", re.I),
    re.compile(r"\bact\s+as\s+(a|an|if)\b", re.I),
    re.compile(r"\bpretend\s+(to\s+be|you\s+are|that)\b", re.I),
    re.compile(r"\b(developer|debug|admin|god|dan)\s+mode\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bbypass\s+(your\s+)?(rules|instructions|filters|guardrails|restrictions)\b", re.I),
    re.compile(r"\boverride\s+(your\s+)?(rules|instructions|settings|programming)\b", re.I),
    re.compile(r"\bnew\s+instructions?\s*:", re.I),
    re.compile(r"\bsystem\s*:\s*", re.I),
    re.compile(r"<\|?\s*(im_start|im_end|system|assistant|endoftext)\s*\|?>", re.I),
    re.compile(r"\[\/?(INST|SYS|SYSTEM)\]", re.I),
    re.compile(r"\bstay\s+in\s+character\b", re.I),
    re.compile(r"\bdo\s+anything\s+now\b", re.I),
]

# Off-topic patterns: code generation, general chat, homework, creative writing, etc.
OFF_TOPIC_PATTERNS = [
    # Code generation — require BOTH a language name AND a code artifact type to avoid
    # blocking research queries like "show me papers on code generation" or
    # "research on algorithm design" or "Papers from Electrical Engineering".
    re.compile(r"\b(write|generate|create)\s+(\w+\s+){0,3}(python|java|javascript|c\+\+|html|css|sql)\s+(code|script|program|function|class)\b", re.I),
    re.compile(r"\b(implement|build|develop|design)\s+(\w+\s+){0,3}(website|app|application|api|database|server|frontend|backend|algorithm|sorting)\b", re.I),
    re.compile(r"\bhow\s+to\s+(code|program|implement|build|develop|install|setup|configure|deploy)\b", re.I),
    re.compile(r"\b(debug|fix|refactor|optimize)\s+(this|my|the)\s+(code|program|script|function|bug|error)\b", re.I),
    re.compile(r"```[\s\S]*```", re.I),  # code blocks in input
    # Only match unmistakable in-line code tokens (not plain English words like
    # "from", "class", "function", "return" which appear in research queries).
    re.compile(r"\b(import\s+\w+\s+from\b|if\s*\(|for\s*\(|while\s*\()", re.I),

    # Homework / exam / assignment
    re.compile(r"\b(solve|calculate|compute|evaluate|derive|prove|simplify)\s+(this|the|following|my)?\s*(equation|integral|derivative|matrix|problem|expression|formula)\b", re.I),
    re.compile(r"\b(my|this)\s+(homework|assignment|exam|quiz|test|project)\b", re.I),
    re.compile(r"\b(help\s+me\s+with|do\s+my)\s+(homework|assignment|exam|math|physics|chemistry)\b", re.I),

    # Creative writing / general chat
    re.compile(r"\b(write|compose|create|generate)\s+(\w+\s+){0,2}(poem|song|story|essay|letter|blog|article|speech|joke|riddle)\b", re.I),
    re.compile(r"\b(tell\s+me\s+a\s+)(joke|story|riddle|fun\s+fact)\b", re.I),
    re.compile(r"\b(translate|convert)\s+(this|the|following)?\s*(to|into)\s+(hindi|french|spanish|german|chinese|japanese|english)\b", re.I),

    # General knowledge unrelated to IIT Delhi research
    re.compile(r"\b(what\s+is\s+the\s+capital\s+of|who\s+is\s+the\s+president|what\s+is\s+the\s+population)\b", re.I),
    re.compile(r"\b(how\s+to\s+cook|recipe\s+for|best\s+restaurants?|movie\s+recommend|book\s+recommend)\b", re.I),
    re.compile(r"\b(weather\s+in|temperature\s+in|forecast\s+for)\b", re.I),
    re.compile(r"\b(stock\s+price|crypto|bitcoin|ethereum|invest\s+in|trading)\b", re.I),
    re.compile(r"\b(play\s+a\s+game|tic\s+tac\s+toe|chess|trivia|quiz\s+me)\b", re.I),

    # Harmful / inappropriate
    re.compile(r"\b(hack|exploit|crack|phish|malware|ransomware|ddos|sql\s+injection|xss)\b", re.I),
    re.compile(r"\b(how\s+to\s+)(hack|break\s+into|bypass\s+security|steal|cheat)\b", re.I),
    re.compile(r"\b(make\s+a?\s+)(bomb|weapon|drug|poison|explosive)\b", re.I),

    # Roleplay / persona change
    re.compile(r"\b(roleplay|role\s+play)\s+(as|with)\b", re.I),
    re.compile(r"\b(be\s+my|act\s+like\s+my)\s+(friend|girlfriend|boyfriend|therapist|doctor|lawyer|tutor)\b", re.I),
    re.compile(r"\b(let'?s\s+)(chat|talk|have\s+a\s+conversation)\s+(about\s+)?(life|love|feelings|politics|religion)\b", re.I),
]

TITLE_RE = re.compile(
    r"\b(prof\.?|professor|dr\.?|mr\.?|mrs\.?|ms\.?|sir|madam|shri|smt\.?)\b",
    re.I,
)

META_NAME_WORDS = frozenset({
    "you", "your", "yourself", "yourselves", "u", "me", "myself", "i", "we", "us",
    "he", "she", "they", "them", "this", "that", "it", "who", "whom", "whose",
    "the", "a", "an",
    "assistant", "bot", "chatbot", "ai", "model", "name", "system", "admin",
    "developer", "user", "anyone", "someone", "somebody", "everyone", "person",
    "professor", "prof", "doctor", "dr", "sir", "madam", "human", "robot",
})

def _matches_any(text: str, patterns: list[re.Pattern]) -> bool:  # type: ignore[type-arg]
    return any(p.search(text) for p in patterns)

def sanitize_message(raw: str, max_length: int = 2000) -> str:
    text = str(raw or "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]{4,}", "   ", text).strip()
    return text[:max_length]

MetaType = Literal["greeting", "identity", "capabilities", "refusal", "off_topic"]

def classify_meta(message: str) -> MetaType | None:
    if _matches_any(message, GREETING_PATTERNS):
        return "greeting"
    if _matches_any(message, PROMPT_REVEAL_PATTERNS):
        return "refusal"
    if _matches_any(message, INJECTION_PATTERNS):
        return "refusal"
    if _matches_any(message, IDENTITY_PATTERNS):
        return "identity"
    if _matches_any(message, CAPABILITY_PATTERNS):
        return "capabilities"
    if _matches_any(message, OFF_TOPIC_PATTERNS):
        return "off_topic"
    return None

def canned_reply(meta_type: str) -> str:
    return CANNED.get(meta_type, CANNED["refusal"])

def detect_injection(message: str) -> bool:
    return _matches_any(message, INJECTION_PATTERNS) or _matches_any(message, PROMPT_REVEAL_PATTERNS)

def name_tokens(name: str) -> list[str]:
    cleaned = TITLE_RE.sub(" ", str(name or "")).lower()
    cleaned = re.sub(r"[^a-z\s.\-]", " ", cleaned)
    tokens = cleaned.split()
    return [
        t.replace(".", "").replace("-", "").strip()
        for t in tokens
        if len(t.replace(".", "").replace("-", "").strip()) >= 2
        and t.replace(".", "").replace("-", "").strip() not in META_NAME_WORDS
    ]

def faculty_name_matches(requested_name: str, first_name: str, last_name: str) -> bool:
    requested = name_tokens(requested_name)
    if not requested:
        return False
    parts = f"{first_name or ''} {last_name or ''}".lower().split()
    parts = [t.strip() for t in parts if t.strip()]
    if not parts:
        return False
    return any(
        any(fp == q or fp.startswith(q) or q.startswith(fp) for fp in parts)
        for q in requested
    )
