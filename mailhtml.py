"""Render one mail's HTML part the way a mail client shows it: shaped, styled,
and unable to execute.

This module is the *only* place that turns attacker-authored HTML into something
the reading pane may show, so everything about it is a decision worth reading:

**Who strips what.** `nh3` (the Rust `ammonia` crate) is the security boundary,
not a helper. It is not the only sanitizer anyone could have picked, and the two
that come up in conversation are both wrong here: `bleach` shipped its final
release in June 2026 and sits on an unmaintained parser, and its defaults — like
every general-purpose comment-section allowlist — have no `img`, no `style`, no
`table`, which is not an HTML mail, that is confetti. The email profile below is
the actual work; the library just has to be a maintained one that parses the way
a browser parses.

**Why a real HTML5 parser matters for safety *and* for fidelity.** `ammonia`
parses with `html5ever`, the WHATWG algorithm: `<td>a<p>x</td>` gets the implied
`<tbody>` and the paragraph the browser would give it, and a whole mail document
gets its `<style>` hoisted out of the `<head>` for free — which is exactly what
Gmail does, and what `html.parser` (used by `mailbox.html_to_text`, for text
extraction, where it is fine) cannot do because it does not build that tree.
A sanitizer whose parse differs from the browser's parse is a sanitizer whose
"what I removed" is not "what the browser sees".

**What the sanitizer cannot do, and what covers it.** `nh3` keeps the text
inside a `<style>` element verbatim: it filters `style` *attributes* through a
property allowlist, but it does not parse stylesheets. So `_narrow_style_blocks`
parses each stylesheet and re-emits it from its own tokens, which buys the one
invariant that matters: no `<` survives inside a `<style>` element, so the CSS
cannot close its own element and become markup. The frame this output goes into
(`frame_document`) is sandboxed with no script permission and an opaque origin,
and carries a CSP that fetches nothing — so a surviving CSS trick could at worst
rearrange the mail inside its own rectangle. That is the same residual risk the
research on mail clients calls out for CSS, and the positioning properties it
names are stripped below, as Gmail strips them.

**Two shapes the guard runs before the sanitizer, not after.** Deeply nested
markup makes `nh3` superlinear (measured here: 20 000 nested divs in 0.8 s,
100 000 in 21.7 s), and one request on a media worker that takes a minute is a
mail that denies service to your own mailbox. Hence `_guard_size` and
`_guard_depth`, which are also the honest reason a pathological mail renders as
text with a sentence saying why.
"""

from __future__ import annotations

import re
from typing import Callable, Optional
from urllib.parse import quote

try:                                    # the security boundary: never optional
    import nh3
except ImportError as e:                # pragma: no cover - import-time only
    raise ImportError(
        "rendering HTML mail needs the 'nh3' package (pip install nh3). "
        "This module is a sanitizer, not a suggestion: without it there is no "
        "safe way to show a mail's HTML, so the reading pane stays text-only.") \
        from e

__all__ = ["MailHtmlError", "MailHtmlTooLarge", "MailHtmlTooDeep",
           "render_email_html", "frame_document", "email_profile",
           "MAX_BYTES", "MAX_DEPTH"]


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

#: A rendered body is a thing a person asked to look at, not a blob. Marketing
#: mail with every logo inlined as base64 gets past a megabyte routinely, so the
#: cap is generous and the refusal it produces is a sentence, not an error page.
MAX_BYTES = 4_000_000

#: Browsers themselves stop nesting at about this depth, and no real mail comes
#: near it — a table-based newsletter nests ten levels, not two hundred. Above
#: it the sanitizer's cost is the problem (see the module docstring), so the
#: markup is refused before it is parsed rather than after it takes a minute.
MAX_DEPTH = 400


class MailHtmlError(Exception):
    """This HTML will not be rendered, for a reason worth showing to a reader."""


class MailHtmlTooLarge(MailHtmlError):
    pass


class MailHtmlTooDeep(MailHtmlError):
    pass


# ---------------------------------------------------------------------------
# The email profile
# ---------------------------------------------------------------------------

#: Tags an HTML mail needs to look like itself. Everything here earns its place
#: with markup that is actually in mailboxes: `font`, `center`, `td` and the
#: `align`/`bgcolor` family are how newsletters are built, and a profile without
#: them does not render email, it renders an apology.
#:
#: Deliberately *not* here, with the reason: `svg`/`math` (re-parse surprises and
#: the `<animate>` CSS-injection class Roundcube fixed), `form`/`input`/`button`
#: (a fake login box is a phishing UI, and Gmail strips them too), `iframe`/
#: `object`/`embed`/`applet` (content from elsewhere), `base` (redirects every
#: other URL in the document — a named sanitizer bypass), `meta`/`link`
#: (refreshes and external stylesheets), `noscript`/`noembed` (a sandboxed frame
#: has no scripting, so a browser *renders* their contents that Gmail treats as
#: inert), `title`/`head`/`html`/`body` as elements (a document's `<title>`
#: otherwise leaks onto the page as visible text), `marquee`/`blink`/`audio`/
#: `video` (motion and media are not what a reading pane is for), and `srcset`
#: — see `_URL_ATTRS`, because `nh3` does not run its scheme allowlist over it.
EMAIL_TAGS = frozenset({
    "a", "abbr", "acronym", "b", "bdi", "bdo", "big", "blockquote", "br",
    "caption", "center", "cite", "code", "col", "colgroup", "dd", "del", "dfn",
    "div", "dl", "dt", "em", "figcaption", "figure", "font", "h1", "h2", "h3",
    "h4", "h5", "h6", "hr", "i", "img", "ins", "kbd", "li", "mark", "ol", "p",
    "pre", "q", "s", "samp", "small", "span", "strike", "strong", "sub", "sup",
    "table", "tbody", "td", "tfoot", "th", "thead", "tr", "tt", "u", "ul", "var",
    "wbr",
    # Kept, then re-emitted from parsed CSS by `_narrow_style_blocks` — never
    # verbatim. This is the head-stylesheet case Gmail honours.
    "style",
})

#: Tags whose *text* goes with them. `style` is in `nh3`'s default list and must
#: stay out of it here, or the mail's stylesheet disappears with the tag; it is
#: in `EMAIL_TAGS` instead. `title` is here because a document's title otherwise
#: arrives as visible text on the page.
DROP_CONTENT_TAGS = frozenset({
    "script", "noscript", "noembed", "template", "iframe", "object", "embed",
    "applet", "title", "head", "xmp", "plaintext",
})

#: URL-bearing attributes, so every one of them passes the scheme check and the
#: per-render image policy in `_attribute_filter`. `background` (a table/td image
#: in old mail markup) is *not* here on purpose: `nh3`'s scheme list covers
#: `href`/`src`/`object[data]`, not it, so whitelisting it would open a remote
#: image the policy cannot see. Same reason `srcset` is not whitelisted: measured
#: here, `srcset="javascript:… 2x"` passes through it untouched.
_URL_ATTRS = frozenset({"href", "src"})

#: Attributes allowed on any tag. `class` because the mail's own stylesheet
#: targets it; `title` for a tooltip; `dir`/`lang` because bidi text breaks
#: without them. Not `data-*`: mail carries tracking in those, and a reading pane
#: has no use for them.
GENERIC_ATTRS = frozenset({"style", "class", "title", "dir", "lang", "id"})

#: Legacy presentation attributes, which is where HTML mail's layout actually
#: lives. Grouped by what they are for rather than by tag, because that is how
#: the profile is audited: drop one and a real newsletter loses its shape.
_TAG_ATTRS: dict[str, set[str]] = {
    "a": {"href", "name", "target", "rel"},
    "img": {"src", "alt", "width", "height", "align", "border", "hspace",
            "vspace", "usemap"},
    "table": {"width", "border", "cellspacing", "cellpadding", "align", "bgcolor",
              "valign", "style", "class"},
    "td": {"width", "height", "align", "valign", "bgcolor", "colspan", "rowspan",
           "nowrap", "border", "scope", "headers"},
    "th": {"width", "height", "align", "valign", "bgcolor", "colspan", "rowspan",
           "nowrap", "scope", "headers"},
    "tr": {"align", "valign", "bgcolor", "height"},
    "tbody": {"align", "valign"}, "thead": {"align", "valign"},
    "tfoot": {"align", "valign"},
    "font": {"face", "size", "color"},
    "div": {"align"}, "p": {"align"}, "hr": {"align", "size", "width", "color"},
    "ul": {"type", "start"}, "ol": {"type", "start"}, "li": {"type"},
    "blockquote": {"cite"}, "col": {"span", "width", "align"},
    "colgroup": {"span", "width", "align"}, "br": {"clear"},
    "del": {"cite"}, "ins": {"cite"}, "bdo": {"dir"}, "abbr": {"title"},
}

#: `style` attribute properties a mail may keep. What mail CSS is made of, plus
#: `display`/`visibility` — which stay allowed because Gmail honours them: a
#: preview-text block written `display:none` must stay invisible here too, or
#: hiding becomes revealing. What is *not* here is the list Gmail strips
#: (`position`, `top`/`right`/`bottom`/`left`, `z-index`, transforms,
#: transitions, animations) — the overlay-and-hide surface the research on mail
#: clients describes — plus anything that could fetch (`behavior`, `filter`) or
#: script (`expression`, `-moz-binding`).
STYLE_PROPERTIES = frozenset({
    "color", "background", "background-color", "background-image",
    "background-position", "background-repeat", "background-size",
    "font", "font-family", "font-size", "font-style", "font-weight",
    "font-variant", "line-height", "letter-spacing", "word-spacing",
    "text-align", "text-decoration", "text-indent", "text-transform",
    "text-overflow", "text-shadow", "vertical-align", "white-space",
    "padding", "padding-top", "padding-right", "padding-bottom", "padding-left",
    "margin", "margin-top", "margin-right", "margin-bottom", "margin-left",
    "border", "border-top", "border-right", "border-bottom", "border-left",
    "border-width", "border-style", "border-color", "border-radius",
    "border-collapse", "border-spacing", "border-top-width", "border-top-style",
    "border-top-color", "border-bottom-width", "border-bottom-style",
    "border-bottom-color", "border-left-width", "border-left-style",
    "border-left-color", "border-right-width", "border-right-style",
    "border-right-color",
    "width", "min-width", "max-width", "height", "min-height", "max-height",
    "display", "float", "clear", "overflow", "overflow-x", "overflow-y",
    "visibility", "opacity", "list-style", "list-style-type", "list-style-image",
    "table-layout", "caption-side", "empty-cells", "box-shadow", "box-sizing",
    "align-items", "align-content", "justify-content", "flex", "flex-basis",
    "flex-direction", "flex-flow", "flex-grow", "flex-shrink", "flex-wrap",
    "grid", "gap", "row-gap", "column-gap", "object-fit", "object-position",
    "-webkit-text-size-adjust", "-webkit-font-smoothing",
})

#: `nh3` checks the scheme of `href`/`src` against this. `cid` is here because
#: inline attachments are addressed that way and `_attribute_filter` is what
#: turns one into a local URL — with `cid` absent the src is stripped before a
#: filter ever sees it (measured). `data` is here only so a *sender's* inlined
#: image can be judged by size in the filter; the placeholder this module
#: injects is its own, and `data:` on a link is refused there.
URL_SCHEMES = frozenset({"http", "https", "mailto", "tel", "cid", "data"})

#: How many bytes of inlined image one `data:` URI may carry before it is
#: replaced. A logo as base64 is normal mail; a 3 MB inlined hero image is a
#: message that did not ask.
MAX_DATA_URI = 300_000

#: 1x1 transparent GIF, 43 bytes. A *blocked* remote image becomes this instead
#: of losing its `src` entirely: the `width`/`height` the sender wrote then still
#: hold the layout open, which is what Gmail's hidden-images state looks like,
#: where a dropped `src` shows a broken-image icon and collapses the table cell.
BLOCKED_IMAGE_PX = (
    "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEA"
    "AAAIBp+nJycKCgoAIf5PY1JPTS4xUFVVVQ=="
)


def email_profile(*, message_url: str = "",
                  cid_map: Optional[dict] = None,
                  images_allowed: bool = False,
                  stats: Optional[dict] = None) -> nh3.Cleaner:
    """The sanitizer this mail is cleaned with, with the image policy baked in.

    The image policy and the `cid:` mapping live in `nh3`'s attribute filter
    rather than in a pass over the output, because the filter sees every
    attribute the sanitizer decided to keep, *before* anything is serialised:
    rewriting URLs after serialisation means parsing HTML again with something
    that is not the browser, which is how mXSS bugs are born.

    A `Cleaner` is built per render for that reason — the closure carries this
    message's id, its cid map, and whether this reader clicked "display images".
    """
    stats = stats if stats is not None else {}
    cid_map = cid_map or {}

    def attribute_filter(tag: str, attr: str, value: str) -> Optional[str]:
        # Links: never a navigation target that outranks the pane.
        if tag == "a" and attr == "target":
            return "_blank"
        if attr not in _URL_ATTRS:
            return value
        low = (value or "").strip().lower()
        if low.startswith("cid:"):
            name = cid_map.get(low[4:].strip().lower())
            if name and message_url:
                stats["inline"] = stats.get("inline", 0) + 1
                return f"{message_url}/{quote(name)}"
            # Gmail shows nothing at all for a cid with no matching part, and
            # keeping the `cid:` src would hand the frame a URL it cannot resolve
            # and a broken-image icon Gmail does not show.
            stats["unresolved"] = stats.get("unresolved", 0) + 1
            return None
        if low.startswith("data:"):
            if len(value) > MAX_DATA_URI:
                stats["oversized_data"] = stats.get("oversized_data", 0) + 1
                return BLOCKED_IMAGE_PX if attr == "src" else None
            if attr == "href":
                # `data:text/html` on a link would load a document of the mail's
                # choosing into the frame, with its own CSS and no relation to
                # the message being read.
                stats["data_link"] = stats.get("data_link", 0) + 1
                return None
            return value
        if low.startswith(("mailto:", "tel:")):
            return value
        if not low.startswith(("http://", "https://")):
            # Relative — and `//host/path`, which is relative in a way that reads
            # as absolute. A sandboxed frame with no same-origin permission still
            # *inherits its host document's base URL*, so `/api/media/x` written
            # inside a mail is a request to this app, and `//host` is a request
            # somewhere else. `url_relative="deny"` cannot be the answer here:
            # measured, it also strips the local URL this filter rewrites a `cid:`
            # into, so the check is made here instead, where the difference
            # between "my URL" and "yours" is still known.
            stats["relative"] = stats.get("relative", 0) + 1
            return None
        if attr == "src":
            if not images_allowed:
                stats["blocked"] = stats.get("blocked", 0) + 1
                return BLOCKED_IMAGE_PX
            stats["shown"] = stats.get("shown", 0) + 1
            return value
        return value

    def style_filter(tag: str, attr: str, value: str) -> Optional[str]:
        """Narrow a `style` attribute to the properties a pane may show.

        The property allowlist is not enough on its own, and the order of the two
        checks is the reason: `nh3` runs `attribute_filter` on the *raw* value, so
        vetoing the whole attribute over one `background:url(...)` would also take
        the `color:red` next to it with it — Gmail keeps the colour. So the
        declarations are filtered one by one here (the same list the stylesheets
        get), and `filter_style_properties` stays on the Cleaner as a second pass
        over whatever survives, since it parses CSS properly and normalises it.
        """
        if attr != "style":
            return value
        kept = _narrow_declarations(value, stats)
        return kept or None

    merged: dict[str, set[str]] = {tag: set(attrs) | GENERIC_ATTRS
                                   for tag, attrs in _TAG_ATTRS.items()}

    return nh3.Cleaner(
        tags=set(EMAIL_TAGS),
        clean_content_tags=set(DROP_CONTENT_TAGS),
        attributes=merged,
        generic_attribute_prefixes=set(),
        # `rel` is set by nh3 itself; passing link_rel would conflict with
        # `rel` being whitelisted above, so the filter forces both instead.
        link_rel=None,
        set_tag_attribute_values={"a": {"target": "_blank",
                                        "rel": "noopener noreferrer"}},
        url_schemes=set(URL_SCHEMES),
        # `deny` would be the natural setting and it is the wrong one here: it is
        # applied *after* the attribute filter, so it also strips the local
        # attachment URL a `cid:` is rewritten into. Relative URLs are refused in
        # the filter instead, where "the URL I made" and "the URL the mail wrote"
        # are still distinguishable.
        url_relative="pass_through",
        filter_style_properties=set(STYLE_PROPERTIES),
        attribute_filter=_Chained(attribute_filter, style_filter),
        strip_comments=True,
    )


class _Chained:
    """Two attribute filters as one, since `nh3` takes a single callback."""

    def __init__(self, *filters: Callable[[str, str, str], Optional[str]]):
        self._filters = filters

    def __call__(self, tag: str, attr: str, value: str) -> Optional[str]:
        for fn in self._filters:
            value = fn(tag, attr, value)
            if value is None:
                return None
        return value


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
         "meta", "param", "source", "track", "wbr"}
_TAG_TOKEN = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)")


def _guard_size(raw: str, max_bytes: int) -> None:
    if len(raw.encode("utf-8", "replace")) > max_bytes:
        raise MailHtmlTooLarge(
            f"this mail's HTML is {len(raw):,} characters, over the "
            f"{max_bytes:,}-byte limit for rendering")


def _guard_depth(raw: str, max_depth: int) -> None:
    """Refuse markup nested deeper than a browser would render it.

    A counter over the tag tokens, not a parse: the point is to cost nothing on
    the way in, and an approximate depth is enough to decide "this is not a
    newsletter" — `nh3` on 100 000 nested divs measured 21.7 s here, on a worker
    whose job is to answer a click in a second.
    """
    depth = best = 0
    for closing, name in _TAG_TOKEN.findall(raw):
        name = name.lower()
        if closing:
            depth = max(0, depth - 1)
        elif name in _VOID:
            continue
        else:
            depth += 1
            if depth > best:
                best = depth
                if best > max_depth:
                    raise MailHtmlTooDeep(
                        f"this mail nests its markup {best} levels deep, past "
                        f"the {max_depth}-level limit for rendering")


# ---------------------------------------------------------------------------
# Stylesheets
# ---------------------------------------------------------------------------

_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)
#: `<!--` and `-->` wrapped around a stylesheet's contents, which Word and Outlook
#: still write. CSS itself ignores them (they are CDO/CDC tokens, and a browser
#: parsing a `<style>` ignores them too), but here an unclaimed `-->` at the end
#: reads as a rule with no block — and the reader would be told a part of the
#: stylesheet was left out when nothing was. They carry no `<` once matched, so
#: removing them only ever removes the thing they are.
_CSS_HTML_COMMENT = re.compile(r"<!--|-->")
#: At-rules whose whole text is dropped. `@import` and `@font-face` fetch (and
#: Gmail removes `@font-face` but its own two fonts); `@keyframes`, `@page` and
#: `@viewport` are motion or print, neither of which a reading pane is for;
#: `@charset` must not survive into a document that is UTF-8 by construction.
_CSS_DROP_AT = ("import", "charset", "font-face", "keyframes", "-webkit-keyframes",
                "namespace", "page", "viewport", "counter-style", "property")
#: At-rules that *contain* rules and so are kept with their prelude filtered.
#: Dropping `@media` would drop the responsive half of a modern newsletter.
_CSS_KEEP_AT = ("media", "supports", "layer", "container")
_BLOCK = re.compile(r"\{")
_STYLE_BLOCK = re.compile(r"(?is)<style\b[^>]*>(.*?)(?:</style>|\Z)")
_BAD_CSS_VALUE = re.compile(r"(?i)(<|url\s*\(|expression\s*\(|javascript:|"
                            r"vbscript:|-moz-binding|@import|behavior\s*:|</)")


def _narrow_stylesheet(css: str, stats: dict) -> str:
    """Re-emit a stylesheet from parsed parts, keeping only what a pane may show.

    The security claim is the reason this exists and it is narrow on purpose:
    the result contains no `<`, so the text of a `<style>` element cannot end
    that element and become markup. Everything else here is fidelity — keeping
    the selectors, `@media` queries and declarations a mail was built from, and
    dropping the properties Gmail drops rather than inventing a stricter client.
    """
    css = _CSS_COMMENT.sub(" ", css or "")
    css = _CSS_HTML_COMMENT.sub(" ", css)
    out: list[str] = []
    out.extend(_narrow_rules(css, stats, depth=0))
    if "css_dropped" not in stats and "css_partial" not in stats:
        stats["css_kept"] = stats.get("css_kept", 0) + len(out)
    return "\n".join(out)


def _narrow_rules(css: str, stats: dict, depth: int) -> list[str]:
    """Statements and rules at one nesting level, in source order."""
    kept: list[str] = []
    i, n = 0, len(css)
    while i < n:
        while i < n and css[i] in " \t\r\n;":
            i += 1
        if i >= n:
            break
        if css[i] != "@":
            end = _find_brace(css, i)
            if end < 0:
                # A selector with no block: trailing junk, and text we cannot
                # attribute to a rule. Dropping it is the honest reading.
                stats["css_partial"] = stats.get("css_partial", 0) + 1
                break
            close = _find_match(css, end)
            if close < 0:
                stats["css_partial"] = stats.get("css_partial", 0) + 1
                break
            sel, body = css[i:end], css[end + 1:close]
            safe = _safe_selector(sel)
            decls = _narrow_declarations(body, stats)
            if safe and decls:
                kept.append(f"{safe}{{{decls}}}")
            i = close + 1
            continue
        # An at-rule. Its prelude ends at a `;` (no block) or at a top-level
        # `{` (a block follows); a `;` *inside* a block terminates the inner
        # rule, not this one, so the prelude is searched only up to the brace.
        brace = _find_top(css, "{", i)
        semi = css.find(";", i)
        if semi < 0 or (brace >= 0 and semi > brace):
            semi = -1                       # no terminator in the prelude
        has_block = brace >= 0 and semi < 0
        name_m = re.match(r"@([-\w-]+)", css[i:])
        name = (name_m.group(1).lower() if name_m else "")
        if name in _CSS_DROP_AT:
            if has_block:
                close = _find_match(css, brace)
                i = n if close < 0 else close + 1
            else:
                i = n if semi < 0 else semi + 1
            stats["css_dropped_at"] = stats.get("css_dropped_at", 0) + 1
            continue
        if name in _CSS_KEEP_AT and has_block:
            close = _find_match(css, brace)
            if close < 0:
                stats["css_partial"] = stats.get("css_partial", 0) + 1
                break
            prelude = _safe_prelude(css[i + 1 + len(name):brace])
            inner = _narrow_rules(css[brace + 1:close], stats, depth + 1)
            if prelude and inner:
                kept.append(f"@{name} {prelude}{{{';'.join(inner)}}}")
            i = close + 1
            continue
        if has_block:
            # An at-rule this pane has no meaning for — `@page`, a margin box, a
            # vendor at-rule. Its own block goes and the sheet carries on:
            # stopping here would drop every rule that follows it, and Word puts
            # `@page WordSection1` in the middle of the stylesheet every mail it
            # sends, which is most of a document nobody meant to withhold.
            close = _find_match(css, brace)
            if close < 0:
                stats["css_partial"] = stats.get("css_partial", 0) + 1
                break
            i = close + 1
        elif semi >= 0:
            i = semi + 1                    # an unknown statement: dropped
        else:
            i = n
        stats["css_dropped_at"] = stats.get("css_dropped_at", 0) + 1
    return kept


def _narrow_declarations(body: str, stats: dict) -> str:
    """`prop:value` pairs the property allowlist accepts, semicolon-joined."""
    out: list[str] = []
    for decl in body.split(";"):
        if ":" not in decl:
            continue
        prop, _, value = decl.partition(":")
        prop, value = prop.strip().lower(), value.strip()
        if prop not in STYLE_PROPERTIES:
            if prop:
                stats["css_props"] = stats.get("css_props", 0) + 1
            continue
        if not value or _BAD_CSS_VALUE.search(value) or "<" in value:
            stats["css_values"] = stats.get("css_values", 0) + 1
            continue
        out.append(f"{prop}:{value}")
    return ";".join(out)


def _safe_selector(sel: str) -> str:
    """Whitespace-normalised selector, or nothing if it carries anything odd.

    `>` stays: it is a combinator, not a risk — a `<` is what ends a `<style>`
    element, and braces or a semicolon here mean the tokenizer and this reader
    disagree about where a rule begins.
    """
    clean = " ".join((sel or "").split())
    if not clean or any(c in clean for c in "<;{}"):
        return ""
    if _BAD_CSS_VALUE.search(clean):
        return ""
    return clean


def _safe_prelude(prelude: str) -> str:
    clean = " ".join((prelude or "").split())
    return "" if not clean or _BAD_CSS_VALUE.search(clean) else clean


def _find_brace(css: str, start: int) -> int:
    m = _BLOCK.search(css, start)
    return m.start() if m else -1


def _find_match(css: str, open_at: int) -> int:
    """Index of the `}` closing the `{` at `open_at`, or -1 when unbalanced."""
    depth, i, n = 0, open_at, len(css)
    while i < n:
        c = css[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _find_top(css: str, needle: str, start: int) -> int:
    """First `needle` outside any `{}` block from `start`, or -1.

    The needle test comes first, or a `{` can never be found by a function
    looking for `{`: counting it as nesting is what the depth is for, but the
    depth only matters for braces already open when the scan arrives.
    """
    depth, i, n = 0, start, len(css)
    while i < n:
        c = css[i]
        if c == needle and depth == 0:
            return i
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                return -1               # closes a block this scan never opened
        i += 1
    return -1


_STYLE_SPAN = re.compile(r"(?is)(<style\b[^>]*>)(.*?)(?:</style>)")


def _narrow_style_blocks(html: str, stats: dict) -> str:
    """Replace every `<style>` element's text with its narrowed re-emission."""

    def repl(m: re.Match) -> str:
        css = _narrow_stylesheet(m.group(2), stats)
        if not css:
            stats["css_empty"] = stats.get("css_empty", 0) + 1
            return ""
        return f"<style>{css}</style>"

    return _STYLE_SPAN.sub(repl, html)


# ---------------------------------------------------------------------------
# The public calls
# ---------------------------------------------------------------------------

def render_email_html(raw: str, *, message_url: str = "",
                      cid_map: Optional[dict] = None,
                      images_allowed: bool = False,
                      max_bytes: int = MAX_BYTES,
                      max_depth: int = MAX_DEPTH) -> dict:
    """A mail's HTML part as a body fragment a sandboxed frame may show.

    `message_url` is the prefix inline attachments are addressed under
    (`/api/message/<id>/attachment`), `cid_map` maps a part's `Content-ID` to the
    filename its attachment URL wants, and `images_allowed` is the reader having
    clicked "display images" for this message — until then a remote image becomes
    a transparent pixel that holds the sender's own width and height.

    Returns `{"html": str, "images_blocked": int, "images_shown": int,
    "inline": int, "notes": [str]}`. `html` is a fragment: the frame document
    around it is `frame_document`'s, never the mail's, which is what keeps the
    CSP and the font defaults out of the sender's reach.
    """
    raw = raw or ""
    if not raw.strip():
        return {"html": "", "images_blocked": 0, "images_shown": 0, "inline": 0,
                "notes": ["this message has no HTML part to show"]}
    _guard_size(raw, max_bytes)
    _guard_depth(raw, max_depth)

    stats: dict = {}
    cleaner = email_profile(message_url=message_url, cid_map=cid_map,
                            images_allowed=images_allowed, stats=stats)
    html = cleaner.clean(raw)
    html = _narrow_style_blocks(html, stats)

    notes: list[str] = []
    if stats.get("blocked"):
        notes.append(f"{stats['blocked']} remote image"
                     f"{'s' if stats['blocked'] != 1 else ''} hidden")
    if stats.get("unresolved"):
        notes.append("an image this message references by cid: is not in it")
    if stats.get("oversized_data"):
        notes.append("an inlined image was too big to draw")
    if stats.get("css_partial"):
        notes.append("part of this message's stylesheet could not be read, "
                     "so it was left out")
    if stats.get("css_dropped_at"):
        notes.append("this message loads a style or font from elsewhere; "
                     "that is refused here")
    return {"html": html,
            "images_blocked": int(stats.get("blocked", 0)),
            "images_shown": int(stats.get("shown", 0)),
            "inline": int(stats.get("inline", 0)),
            "notes": notes}


#: The frame's own document. Everything a mail is allowed to look like, and
#: nothing it is allowed to do: `default-src 'none'` fetches nothing at all, so
#: a stylesheet rule that still names a URL cannot reach it, and `img-src` only
#: widens to the network when this reader asked for images.
_CSP_BLOCKED = ("default-src 'none'; img-src 'none'; style-src 'unsafe-inline'; "
                "form-action 'none'; base-uri 'none'")
_CSP_ALLOWED = ("default-src 'none'; img-src http: https: data:; "
                "style-src 'unsafe-inline'; form-action 'none'; base-uri 'none'")

_FRAME_STYLE = """
html,body{margin:0;padding:0;background:#fff;color:#000;
  font:13px/1.45 "Helvetica Neue",Helvetica,Arial,sans-serif;
  -webkit-text-size-adjust:100%;word-wrap:break-word;}
body{padding:14px 16px 22px;}
a{color:#1a0dab;text-decoration:none;}
a:visited{color:#609;}
a:hover{text-decoration:underline;}
img{border:0;}
table{border-collapse:separate;}
blockquote{margin:1em 0 1em .8em;padding-left:.8em;border-left:2px solid #dadce0;}
pre{white-space:pre-wrap;word-break:break-word;}
.mbox-blocked{background:#f1f3f4;color:#5f6368;}
""".strip()


def frame_document(body_html: str, *, images_allowed: bool = False) -> str:
    """A whole document for `iframe.srcdoc`, built here and not by the mail.

    The CSP and the font defaults are in *this* template, spliced around the
    sanitized fragment, so no mail content is ever in the string before the
    policy is written — a sender cannot negotiate with the CSP of the document
    they are being placed inside.
    """
    csp = _CSP_ALLOWED if images_allowed else _CSP_BLOCKED
    return (
        "<!DOCTYPE html><html><head>"
        '<meta http-equiv="Content-Type" content="text/html; charset=utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{csp}">'
        f"<style>{_FRAME_STYLE}</style>"
        "</head><body>"
        f'{body_html}'
        "</body></html>"
    )

