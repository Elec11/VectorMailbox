"""Turn 'facturen van flexa van vorige week met bijlage' into structured filters.

Stdlib only, and it must not import mailbox.py: this module is read by the
search path, by the eval harness, and by tests, and a dataclass of filters must
never depend on a database connection or on numpy to be understood.

Why this exists: 'from:flexa factuur van vorige week' embedded as one sentence
makes the vector drift toward the words 'van' and 'week', which is not what the
user meant. Sender, recipient, date and attachment are *constraints*, not
content - so they are parsed out, applied as SQL before similarity search, and
removed from the text the embedder sees.

Supported shapes (Dutch and English, because the mailbox is Dutch):

    from:anna            van:anna            van anna            mail van anna
    to:me / aan:me / aan iem@x.nl           cc: - not supported
    subject:offerte      onderwerp:offerte
    after:2024-05-01     before:12 mei       sinds 1 mei   tot 12-5-2024
    vandaag gisteren deze.week vorige.week deze.maand vorige.maand
    vorig.jaar dit.jaar vorig kwartaal vorige maand
    in maart / maart 2024 / Q1 2024 / 2024
    3 dagen geleden   ouder dan 2 weken   laatste 3 maanden
    met bijlage       with attachment     has:attachment   heeft bijlagen
    is:unread         ongelezen
    larger:1m         - not supported (attachment size is not indexed)

Everything that is not a filter stays in `residual`, which is what gets
embedded. An unparsed 'van' is worse for recall than a missing filter, so every
rule here errs toward leaving the words in the query text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

__all__ = ["Filters", "parse", "describe"]

# Words that end a bare 'van <naam>' phrase: 'factuur van flexa over de factuur'
# must not read 'over' as part of the sender's name.
_STOP = {
    "de", "het", "een", "en", "of", "van", "voor", "met", "over", "aan", "op",
    "in", "bij", "na", "sedert", "sinds", "per", "the", "a", "an", "and", "or",
    "for", "with", "about", "to", "at", "of", "from", "that", "this",
    # Pronouns. 'Statusupdate van je order' names no sender: a bare 'van' in a
    # sentence is a preposition far more often than it is a sender, and every
    # wrong sender filter throws away the whole result set.
    "je", "jij", "jou", "jouw", "u", "uw", "uwing", "ik", "mij", "mijn", "me",
    "we", "wij", "ons", "onze", "hij", "haar", "hun", "hen", "zij", "ze", "men",
    "iemand", "niemand", "iedereen", "allemaal",
}

MONTHS = {
    "januari": 1, "februari": 2, "maart": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "augustus": 8, "september": 9, "oktober": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mrt": 3, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "okt": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april ": 4, "may": 5, "june": 6,
    "july": 7, "october": 10, "december ": 12,
}
WEEKDAYS = {
    "maandag": 0, "dinsdag": 1, "woensdag": 2, "donderdag": 3, "vrijdag": 4,
    "zaterdag": 5, "zondag": 6,
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
    "saturday": 5, "sunday": 6,
}
UNIT_DAYS = {
    "dag": 1, "dag(en)": 1, "dagen": 1, "day": 1, "days": 1,
    "week": 7, "weken": 7, "weeks": 7,
    "maand": 30, "maanden": 30, "month": 30, "months": 30,
    "kwartaal": 91, "kwartalen": 91, "quarter": 91, "quarters": 91,
    "jaar": 365, "jaren": 365, "year": 365, "years": 365,
}


@dataclass
class Filters:
    """Structured constraints parsed out of a query. None/empty = no constraint."""
    senders: list[str] = field(default_factory=list)
    recipients: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    since: Optional[str] = None          # inclusive ISO date, 'YYYY-MM-DD'
    until: Optional[str] = None          # exclusive ISO date
    has_attachment: Optional[bool] = None
    unread: Optional[bool] = None
    residual: str = ""                   # what is left to embed
    notes: list[str] = field(default_factory=list)   # what could not be honoured

    def is_empty(self) -> bool:
        return not (self.senders or self.recipients or self.subjects
                    or self.since or self.until
                    or self.has_attachment is not None
                    or self.unread is not None)

    def as_dict(self) -> dict:
        return {
            "senders": self.senders, "recipients": self.recipients,
            "subjects": self.subjects, "since": self.since, "until": self.until,
            "has_attachment": self.has_attachment, "unread": self.unread,
            "residual": self.residual, "notes": self.notes,
        }


def describe(f: Filters) -> str:
    """One line saying what was filtered out of the query ('' when nothing)."""
    bits = []
    if f.senders:
        bits.append("van " + ", ".join(f.senders))
    if f.recipients:
        bits.append("aan " + ", ".join(f.recipients))
    if f.subjects:
        bits.append("onderwerp " + ", ".join(f.subjects))
    if f.since or f.until:
        bits.append(f"{f.since or '…'} tot {f.until or '…'}")
    if f.has_attachment:
        bits.append("met bijlage")
    if f.unread:
        bits.append("ongelezen")
    bits += list(f.notes)
    return " · ".join(bits)


def _month_word() -> str:
    return "|".join(sorted((m for m in MONTHS if m.strip() == m), key=len, reverse=True))


def _iso(d: date) -> str:
    return d.isoformat()


def _first_of_month(d: date, back: int = 0) -> date:
    m = d.month - 1 - back
    y, m = (d.year + (m // 12)) if m < 0 else d.year, m % 12 + 1
    if m == 0:                            # back crossed a year boundary
        y, m = y - 1, 12
    return date(y, m, 1)


def _quarter_start(d: date, back: int = 0) -> date:
    q = (d.month - 1) // 3 - back
    y = d.year + (q // 4) if q >= 0 else d.year - 1
    q = q % 4
    return date(y, q * 3 + 1, 1)


def _date_patterns() -> list[tuple[str, str]]:
    """(regex, kind) pairs, longest phrase first - order is the parse priority."""
    mo = _month_word()
    return [
        # --- relative words -------------------------------------------------
        (r"\bvandaag\b|\btoday\b", "today"),
        (r"\bgisteren\b|\byesterday\b", "yesterday"),
        (r"\bdeze\s+week\b|\bthis\s+week\b|\bdeze\s+w\b", "this_week"),
        (r"\bvorige\s+week\b|\blast\s+week\b|\bafgelopen\s+week\b"
         r"|\blaatste\s+week\b", "last_week"),
        (r"\bprevious\s+(week|month|year)\b", "previous_n"),
        (r"\bdeze\s+maand\b|\bthis\s+month\b", "this_month"),
        (r"\bvorige\s+maand\b|\blast\s+month\b|\bafgelopen\s+maand\b"
         r"|\blaatste\s+maand\b", "last_month"),
        (r"\bvorig(?:e)?\s+kwartaal\b|\blaatste\s+kwartaal\b"
         r"|\blast\s+quarter\b|\bprevious\s+quarter\b", "last_quarter"),
        (r"\bthis\s+quarter\b|\bdit\s+kwartaal\b", "this_quarter"),
        (r"\bvorig(?:e)?\s+jaar\b|\blaatste\s+jaar\b|\blast\s+year\b",
         "last_year"),
        (r"\bdit\s+jaar\b|\bthis\s+year\b|\bcurrent\s+year\b", "this_year"),
        (r"\brecent\b|\brecentelijk\b|\bnieuw\b", "recent"),
        (r"\bold\b|\boude\b|\bolder\b", "old"),
        # --- counted --------------------------------------------------------
        (r"\b(?:laatste|affgelopen|afgelopen|past|last)\s+(\d+)\s+"
         r"(dagen|dag|weken|week|maanden|maand|jaren|jaar|days|day|weeks|week"
         r"|months|month|years|year)\b", "last_n"),
        (r"\b(\d+)\s+(dagen|dag|weken|week|maanden|maand|jaren|jaar|days|day"
         r"|weeks|week|months|month|years|year)\s+geleden\b", "ago_n"),
        (r"\b(ouder|older)\s+dan\s+(\d+)\s+(dagen|dag|weken|week|maanden|maand"
         r"|jaren|jaar|days|day|weeks|week|months|month|years|year)\b", "older_than"),
        (r"\b(recent|jong|newer|nieuwer)\s+dan\s+(\d+)\s+(dagen|dag|weken|week"
         r"|maanden|maand|jaren|jaar|days|day|weeks|week|months|month|years|year)\b",
         "newer_than"),
        # --- explicit ranges ------------------------------------------------
        (rf"\bsinds\s+(?P<a1>\d{{4}}-\d{{2}}-\d{{2}})\b", "since_iso"),
        (rf"\b(?:sinds|sedert|from|after|na)\s+(?P<a>\d{{1,2}}[/-]\d{{1,2}}"
         rf"(?:[/-]\d{{2,4}})?\b)", "since_dmy"),
        (rf"\b(?:sinds|sedert|from|after)\s+((?:{mo})\s+\d{{4}})\b", "since_month"),
        # 'tot'/'voor' with an ISO date is the upper bound, not a day to search:
        # without this the ISO below is read as a single date and 'voor 2026-03
        # -05' returns the mail *of* 5 March, i.e. nothing.
        (rf"\b(?:tot|t/m|tm|before|until|voor)\s+(?P<b1>\d{{4}}-\d{{2}}-\d{{2}})\b",
         "until_iso"),
        (rf"\b(?:tot|t/m|tm|before|until|voor)\s+(?P<b>\d{{1,2}}[/-]\d{{1,2}}"
         rf"(?:[/-]\d{{2,4}})?)\b", "until_dmy"),
        (rf"\b(?:tussen|between)\s+(?P<t1>\S+)\s+(?:en|and)\s+(?P<t2>\S+)\b", "between"),
        # --- single dates ---------------------------------------------------
        (r"\b(20\d{2})-(\d{2})-(\d{2})\b", "ymd"),
        (rf"\b(\d{{1,2}})[/-](\d{{1,2}})(?:[/-](\d{{2,4}}))?\b", "dmy"),
        # '5 mei 2024' must be read before 'mei 2024': the more specific phrase
        # wins, otherwise the day is left in the query and a search for one day
        # quietly becomes a search for the whole month.
        (rf"\b(\d{{1,2}})\s+({mo})(?:\s+(20\d{{2}}))?\b", "day_month"),
        (rf"\b({mo})\s+(20\d{{2}}|2\d)\b", "month_year"),
        (rf"\b(?:in|in\s+de\s+maand)\s+({mo})\b(?:\s+(20\d{{2}}))?", "month_only"),
        (r"\bq([1-4])\s*(20\d{2})?\b", "quarter"),
        (r"\b(20\d{2})\b", "year"),
    ]


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip(" ,.;:!?'\"")).strip()


def _quoted_or_words(text: str, i: int) -> tuple[str, int]:
    """Read one filter value at `i`: a "quoted" run, an e-mail, or 1-2 words.

    Two words is the whole budget ('jan jansen', not 'jan jansen over de
    offerte'), and a token that looks like a date or a month never starts a
    name - 'from:anna 12-5-2024' must not read the date into the sender.
    """
    m = re.match(r'"([^"]+)"', text[i:])
    if m:
        return m.group(1), i + m.end()
    m = re.match(r"[^\s,;]+@[^\s,;]+", text[i:])
    if m:
        return m.group(0), i + m.end()
    words, out = [], i
    for tok in re.finditer(r"\S+", text[i:]):
        w = tok.group(0).strip(".,;:!?")
        low = w.lower()
        if (not w or low in _STOP or len(words) >= 2
                or re.match(r"^\d", w)
                or re.match(r"^(from|van|to|naar|aan|subject|onderwerp|has|met"
                            r"|after|before|sinds|tot|is|in|vorig|vorige|deze"
                            r"|vandaag|gisteren|last|previous|this)$", low)):
            break
        words.append(w)
        out = i + tok.end()
    return " ".join(words), out


def _looks_like_name(value: str) -> bool:
    """Guard for the bare `van X` form: 'van de factuur' is not a sender.

    'jan' is both a Dutch first name and the abbreviation of January, so case
    settles it: 'van Jan' is a person, 'van maart' is a month. A real month
    phrase ('maart 2024', 'vorige maand') has already claimed its span before
    this runs, so it never reaches here.
    """
    v = _clean(value)
    if not v:
        return False
    if "@" in v:
        return True
    parts = v.split()
    if len(parts) > 2 or any(p.lower() in _STOP for p in parts):
        return False
    for p in parts:
        if re.match(r"^\d", p):
            return False
        # Only a lone month word is a month: 'jan jansen' is a person, and a
        # real month phrase already claimed its span before this runs.
        if len(parts) == 1 and p.lower() in MONTHS and not p[:1].isupper():
            return False
    if len(v) < 3:
        return False
    return bool(re.match(r"^[\w][\w .'\-@]+$", v))


def parse(query: str, today: Optional[date] = None) -> Filters:
    """Extract Gmail-style and natural-language filters from `query`.

    `today` is a parameter, not datetime.now(): the date maths is the part of
    this module that is easiest to get wrong and must be testable.
    """
    text = " ".join((query or "").split())
    today = today or date.today()
    f = Filters(residual=text)
    spans: list[tuple[int, int]] = []

    def take(a: int, b: int) -> None:
        spans.append((a, b))

    # -- explicit key:value, both languages ---------------------------------
    kv = re.compile(
        r"(?i)\b(from|van|afzender|sender|to|aan|ontvanger|subject|onderwerp|"
        r"has|heeft|is|larger|na|after|before|sinds|tot|sedert|until|voor)\s*:\s*",
    )
    pos, guard = 0, 0
    while guard < 24:
        guard += 1
        m = kv.search(text, pos)
        if not m:
            break
        key = m.group(1).lower()
        value, end = _quoted_or_words(text, m.end())
        if not value:
            pos = m.end()
            continue
        take(m.start(), end)
        if key in ("from", "van", "afzender", "sender"):
            f.senders.append(value)
        elif key in ("to", "aan", "ontvanger"):
            f.recipients.append(value)
        elif key in ("subject", "onderwerp"):
            f.subjects.append(value)
        elif key in ("has", "heeft"):
            if value.lower() in ("attachment", "attachments", "bijlage", "bijlagen",
                                 "bijlage:n", "file", "files"):
                f.has_attachment = True
            elif value.lower() in ("unread", "ongelezen"):
                f.unread = True
            elif value.lower() in ("image", "afbeelding", "screenshot", "foto"):
                f.has_attachment = True
            else:
                f.notes.append(f"has:{value} is not indexed")
        elif key == "is":
            if value.lower() in ("unread", "ongelezen", "unread"):
                f.unread = True
            elif value.lower() in ("read", "gelezen"):
                f.unread = False
            else:
                f.notes.append(f"is:{value} is not supported")
        elif key in ("larger",):
            f.notes.append("attachment size (larger:) is not indexed")
        elif key in ("after", "na", "sinds", "from", "sedert"):
            d = _parse_date(value, today)
            if d:
                f.since = _iso(d)
        elif key in ("before", "voor", "tot", "until"):
            d = _parse_date(value, today)
            if d:
                f.until = _iso(d + timedelta(days=1))   # 'tot X' includes X
        pos = end

    # -- bare 'met bijlage(n)' / 'with attachment(s)' ------------------------
    for m in re.finditer(
            r"(?i)\b(?:met|having|having|with|having|heeft|has)\s+"
            r"(?:een|one)?\s*(?:bijlage|bijlagen|attachment|attachments|file|files|"
            r"pdf|excel|word|factuur\.pdf)\b", text):
        take(m.start(), m.end())
        f.has_attachment = True

    # -- bare 'ongelezen' / 'unread' -----------------------------------------
    for m in re.finditer(r"(?i)\b(?:ongelezen|unread)\b", text):
        take(m.start(), m.end())
        f.unread = True

    # -- dates ---------------------------------------------------------------
    for pattern, kind in _date_patterns():
        for m in re.finditer(pattern, text, re.I):
            a, b = m.span()
            if any(a < y and b > x for x, y in spans):
                continue                      # inside an already-taken phrase
            res = _apply_date(kind, m, today, f)
            if res:
                take(a, b)

    # -- bare 'van <naam>' / 'aan <naam>' ------------------------------------
    # Runs last, after dates took their phrases, and only on values that do not
    # overlap an already-taken span: a false sender filter throws away the whole
    # result set, a missed one only loses a little precision.
    def free(a: int, b: int) -> bool:
        return not any(a < y and b > x for x, y in spans)

    for m in re.finditer(r"(?iu)\b(?:van|from|mail\s+van|afzender|sender)\s+", text):
        if not free(m.start(), m.end()):
            continue
        value, end = _quoted_or_words(text, m.end())
        value = _clean(value)
        if not value or not free(m.end(), end) or not _looks_like_name(value):
            continue
        if re.search(r"(?i)\b(datum|dag|jaar|maand|week|kwartaal|prijs|bedrag"
                     r"|date|year|month|week|amount|price|version)\b", value):
            continue                          # 'van vorige week' was taken above;
        f.senders.append(value)               # 'van de factuur' is not a sender
        take(m.start(), end)
    for m in re.finditer(r"(?iu)\b(?:aan|naar|to|ontvanger|recipient)\s+", text):
        if not free(m.start(), m.end()):
            continue
        value, end = _quoted_or_words(text, m.end())
        value = _clean(value)
        if "@" not in value and not _looks_like_name(value):
            continue
        if not free(m.end(), end):
            continue
        if re.search(r"(?i)\b(mij|me|ons|us|allemaal|iedereen|myself|you|your"
                     r"|date|time|day|today|tomorrow|morning|evening|speed"
                     r"|terms|order|question)\b", value):
            continue                          # 'aan mij' cannot be filtered, and
                                              # 'up to date' is not a recipient
        f.recipients.append(value)
        take(m.start(), end)

    f.residual = _residual(text, spans)
    if f.unread is not None:
        f.notes.append("unread/ongelezen: IMAP flags are not indexed, ignored")
        f.unread = None
    return f


def _apply_date(kind: str, m: re.Match, today: date, f: Filters) -> bool:
    """Apply one matched date phrase. False = leave the words in the query."""
    g = m.groups()
    if kind == "today":
        f.since, f.until = _iso(today), _iso(today + timedelta(days=1))
    elif kind == "yesterday":
        f.since, f.until = _iso(today - timedelta(days=1)), _iso(today)
    elif kind == "this_week":
        start = today - timedelta(days=today.weekday())
        f.since, f.until = _iso(start), _iso(today + timedelta(days=1))
    elif kind == "last_week":
        start = today - timedelta(days=today.weekday() + 7)
        f.since, f.until = _iso(start), _iso(start + timedelta(days=7))
    elif kind == "this_month":
        f.since, f.until = _iso(_first_of_month(today)), _iso(today + timedelta(days=1))
    elif kind == "last_month":
        start = _first_of_month(today, 1)
        f.since, f.until = _iso(start), _iso(_first_of_month(today))
    elif kind == "last_quarter":
        start = _quarter_start(today, 1)
        f.since, f.until = _iso(start), _iso(_quarter_start(today))
    elif kind == "this_quarter":
        start = _quarter_start(today)
        f.since, f.until = _iso(start), _iso(today + timedelta(days=1))
    elif kind == "previous_n":
        what = (m.group(1) or "").lower()
        if what == "week":
            start = today - timedelta(days=today.weekday() + 7)
            f.since, f.until = _iso(start), _iso(start + timedelta(days=7))
        elif what == "month":
            f.since, f.until = _iso(_first_of_month(today, 1)), \
                _iso(_first_of_month(today))
        else:
            f.since, f.until = f"{today.year - 1}-01-01", f"{today.year}-01-01"
    elif kind == "last_year":
        f.since, f.until = f"{today.year - 1}-01-01", f"{today.year}-01-01"
    elif kind == "this_year":
        f.since, f.until = f"{today.year}-01-01", _iso(today + timedelta(days=1))
    elif kind == "recent":
        f.since = _iso(today - timedelta(days=30))
    elif kind == "old":
        f.until = _iso(today - timedelta(days=180))
    elif kind in ("last_n", "ago_n", "newer_than"):
        n, unit = _count_unit(m)
        if not n:
            return False
        if kind == "newer_than":
            f.since = _iso(today - timedelta(days=n * unit))
        else:
            f.since, f.until = _iso(today - timedelta(days=n * unit)), \
                _iso(today + timedelta(days=1))
    elif kind == "older_than":
        n, unit = _count_unit(m)
        if not n:
            return False
        f.until = _iso(today - timedelta(days=n * unit))
    elif kind in ("ymd", "dmy", "month_year", "day_month", "month_only", "year",
                  "since_iso", "since_dmy", "since_month", "until_iso", "until_dmy",
                  "between", "quarter"):
        return _apply_explicit(kind, m, today, f)
    else:
        return False
    return True


def _count_unit(m: re.Match) -> tuple[int, int]:
    nums = [g for g in m.groups() if g and g.isdigit()]
    unit = None
    for g in m.groups():
        if g and g.lower() in UNIT_DAYS:
            unit = UNIT_DAYS[g.lower()]
    if not nums or unit is None:
        return 0, 0
    return int(nums[0]), unit


def _parse_date(value: str, today: date) -> Optional[date]:
    v = _clean(value).lower()
    if not v:
        return None
    if v in ("vandaag", "today"):
        return today
    if v in ("gisteren", "yesterday"):
        return today - timedelta(days=1)
    m = re.fullmatch(r"(20\d{2})-(\d{1,2})-(\d{1,2})", v)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", v)
    if m:
        y = _year(m.group(3), today)
        return _safe_date(y, int(m.group(2)), int(m.group(1)))
    m = re.fullmatch(rf"(\d{{1,2}})\s+({_month_word()})\s+(20\d{{2}})", v)
    if m:
        return _safe_date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    m = re.fullmatch(rf"(\d{{1,2}})\s+({_month_word()})", v)
    if m:
        return _safe_date(today.year, MONTHS[m.group(2)], int(m.group(1)))
    m = re.fullmatch(rf"({_month_word()})\s+(\d{{4}})", v)
    if m:
        return _safe_date(int(m.group(2)), MONTHS[m.group(1)], 1)
    m = re.fullmatch(rf"({_month_word()})", v)
    if m:
        return _safe_date(today.year, MONTHS[m.group(1)], 1)
    m = re.fullmatch(r"(20\d{2})", v)
    if m:
        return _safe_date(int(m.group(1)), 1, 1)
    return None


def _year(text: Optional[str], today: date) -> int:
    if not text:
        return today.year
    y = int(text)
    if y < 100:
        y += 2000 if y <= today.year % 100 else 1900
    return y


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _apply_explicit(kind: str, m: re.Match, today: date, f: Filters) -> bool:
    g = [x for x in m.groups() if x]
    if kind in ("since_iso", "until_iso", "until_dmy", "since_dmy", "since_month",
                "ymd", "dmy",
                "month_year", "day_month", "month_only", "year", "quarter"):
        if kind == "year":
            year = int(g[0])
            start, end = date(year, 1, 1), date(year + 1, 1, 1)
        elif kind == "quarter":
            q = int(m.group(1))
            year = int(g[-1]) if len(g) > 1 else today.year
            start = date(year, (q - 1) * 3 + 1, 1)
            end = date(year + (q == 4), 1 if q == 4 else q * 3 + 1, 1)
        else:
            # 'sinds <x>' / 'tot <x>' must parse the VALUE, not the whole match:
            # _parse_date('sinds 2025-01-15') is None, and a silently unparsed
            # 'sinds' then leaks into the embedded query text.
            raw = g[-1] if kind.startswith(("since", "until")) else m.group(0)
            d = _parse_date(raw, today)
            if d is None:
                return False
            if kind in ("month_year", "month_only"):
                start = date(d.year, d.month, 1)
                end = date(d.year + (d.month == 12), 1 if d.month == 12 else d.month + 1, 1)
            else:
                start = end = d
        if kind.startswith("since"):
            f.since = _iso(start)
        elif kind.startswith("until"):
            f.until = _iso(end + timedelta(days=1))
        elif f.since and f.until:
            return False
        else:
            # For a whole period (year/quarter/month) `end` is already the
            # first day of the next one; adding a day to that would let 'Q1
            # 2025' also return 1 April. A single day has start == end, and
            # there the day after is the exclusive bound.
            period = kind in ("year", "quarter", "month_year", "month_only")
            f.since, f.until = _iso(start), _iso(end if period
                                               else end + timedelta(days=1))
        return True
    if kind == "between":
        a = _parse_date(g[0] if len(g) > 0 else "", today)
        b = _parse_date(g[1] if len(g) > 1 else "", today)
        if not a or not b:
            return False
        f.since, f.until = _iso(a), _iso(b + timedelta(days=1))
        return True
    return False


def _residual(text: str, spans: list[tuple[int, int]]) -> str:
    """Query text with every filter span blanked out, then tidied."""
    out, last = [], 0
    for a, b in sorted(spans):
        if a < last:
            continue
        out.append(text[last:a])
        last = max(last, b)
    out.append(text[last:])
    left = " ".join("".join(out).split())
    # Dangling connectives: 'facturen van' after 'van flexa' was taken elsewhere.
    left = re.sub(r"(?i)\b(?:van|from|met|with|has|heeft|voor|for|in|to|aan)\s*$",
                  "", left).strip()
    left = re.sub(r"(?i)^(?:van|from|met|with|de|het|the)\s+", "", left).strip()
    return left
