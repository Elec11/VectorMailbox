#!/usr/bin/env python3
"""
Tests for the reading pane's sanitizer: what a mail may look like, and what it
may never do.

These are the tests where a passing run means something. A sanitizer is not
verified by checking that good input survives — it is verified by checking that
the specific bypass classes are dead in *this* build, with the payloads that were
used against real mail clients: an attribute that is a URL the scheme list does
not read (`srcset`), a comment that a naive cleaner unwraps into markup, a
`<noscript>` element a script-less frame *renders* instead of ignoring, a
`</style>` inside a CSS string, a relative URL that resolves against the reading
pane's own origin because a sandboxed frame inherits its host's base URL.

   python test_mailhtml.py
"""
from __future__ import annotations

import re
import time
import unittest

try:
    import mailhtml
    HAVE_NH3 = True
except ImportError:                                     # pragma: no cover
    mailhtml = None
    HAVE_NH3 = False

NEEDS_NH3 = "nh3 is the sanitizer; without it there is nothing to test"

STYLE_SPANS = re.compile(r"(?is)<style\b[^>]*>(.*?)</style>")
MSG_URL = "/api/message/ABC123/attachment"


def render(html: str, **kw) -> dict:
    kw.setdefault("message_url", MSG_URL)
    return mailhtml.render_email_html(html, **kw)


def body(html: str, **kw) -> str:
    return render(html, **kw)["html"]


@unittest.skipUnless(HAVE_NH3, NEEDS_NH3)
class ScriptNeverRuns(unittest.TestCase):
    """The frame has no script permission, and nothing script-shaped arrives in
    it in the first place: the two defences are independent on purpose."""

    def test_tags_and_handlers_that_execute_are_gone(self):
        html = body('<p>a</p><script>document.title="owned"</script>'
                    '<img src="http://e/x.png" onerror="fetch(\'/api/leak\')">'
                    '<div onclick="steal()" onload=x onMouseOver=y>z</div>'
                    '<iframe srcdoc="&lt;script&gt;alert(1)&lt;/script&gt;"></iframe>'
                    '<object data="http://e/f.swf"></object>'
                    '<form action="http://e/p"><input name=u><button>Log in</button></form>')
        low = html.lower()
        for needle in ("script", "onerror", "onclick", "onload", "onmouseover",
                       "iframe", "object", "form", "input", "swf"):
            self.assertNotIn(needle, low, f"{needle} survived: {html}")
        self.assertIn("z", html)

    def test_url_schemes_that_execute_are_gone(self):
        html = body('<a href="javascript:alert(1)">js</a>'
                    '<a href="JaVaScRiPt&colon;alert(2)">js2</a>'
                    '<a href="vbscript:msgbox(3)">vbs</a>'
                    '<a href="data:text/html,<b>doc</b>">doc</a>'
                    '<a href="http://ok/x">ok</a>')
        low = html.lower()
        self.assertNotIn("javascript", low)
        self.assertNotIn("vbscript", low)
        self.assertNotIn("data:", low)
        self.assertIn('href="http://ok/x"', html)
        self.assertIn("js", html)                       # the words stay, the URL goes

    def test_srcset_is_not_whitelisted_at_all(self):
        """`nh3` checks the scheme of `href`/`src`; `srcset` is a list of URLs it
        does not read, so whitelisting it would open a URL the policy cannot see.
        """
        html = body('<img src="http://e/a.png" srcset="http://e/b.png 2x">',
                    images_allowed=True)
        self.assertNotIn("srcset", html)
        self.assertIn('src="http://e/a.png"', html)

    def test_base_href_cannot_retarget_the_document(self):
        html = body('<base href="http://evil/"><p>a</p><a href="http://ok/x">l</a>')
        self.assertNotIn("<base", html.lower())
        self.assertIn('href="http://ok/x"', html)

    def test_refresh_and_external_stylesheets_do_not_survive(self):
        html = body('<meta http-equiv="refresh" content="0;url=http://evil/">'
                    '<link rel="stylesheet" href="http://evil/x.css">'
                    '<link rel="import" href="http://evil/i.html"><p>text</p>')
        low = html.lower()
        self.assertNotIn("refresh", low)
        self.assertNotIn("<meta", low)
        self.assertNotIn("<link", low)
        self.assertIn("text", html)

    def test_noscript_content_is_not_shown(self):
        """The frame has no scripting, so a browser *renders* `<noscript>`
        children. Gmail treats that content as inert, so it must not arrive:
        an escape stylesheet or a payload inside a `<noscript>` is a live one here.
        """
        html = body('<p>a</p><noscript><style>body{display:none}</style>'
                    '<img src=x onerror=alert(1)><p>FREE MONEY</p></noscript>')
        self.assertNotIn("FREE MONEY", html)
        self.assertNotIn("onerror", html)
        self.assertNotIn("noscript", html.lower())

    def test_svg_is_out_of_the_profile(self):
        # Roundcube fixed an SVG `<animate>` that injected CSS through an
        # attribute; the shape of the bug is "foreign content parses differently",
        # and mail does not need SVG to be legible.
        html = body('<svg><animate attributeName="fill" to="red"/>'
                    '<script>alert(1)</script><text>svg text</text></svg><p>p</p>')
        self.assertNotIn("svg", html.lower())
        self.assertNotIn("animate", html.lower())
        self.assertIn("p", html)


@unittest.skipUnless(HAVE_NH3, NEEDS_NH3)
class CommentsAndParsing(unittest.TestCase):
    def test_outlook_conditional_comments_stay_invisible(self):
        """Unwrapping a comment is the classic cleaner bug: the text inside
        `<!--[if mso]>…-->` becomes live markup the moment it is uncommented.
        """
        html = body('<p>a</p><!--[if mso]><table><tr><td>OUTLOOK ONLY</td></tr>'
                    '</table><![endif]--><!-- <img src=x onerror=alert(1)> -->')
        self.assertNotIn("OUTLOOK ONLY", html)
        self.assertNotIn("onerror", html)
        self.assertNotIn("<!--", html)

    def test_downlevel_revealed_content_is_kept(self):
        # `<!--[if !mso]><!--> x <!--<![endif]-->` is written for non-Outlook
        # readers, and this pane is a non-Outlook reader.
        html = body('<p>a</p><!--[if !mso]><!--><span>for modern mail</span>'
                    '<!--<![endif]-->')
        self.assertIn("for modern mail", html)

    def test_a_documents_title_does_not_leak_as_text(self):
        html = body('<html><head><title>Invoice 4417</title></head>'
                    '<body><p>real body</p></body></html>')
        self.assertIn("real body", html)
        self.assertNotIn("Invoice 4417", html)

    def test_the_table_the_parser_builds_is_the_table_the_browser_builds(self):
        # Implied `tbody`/`p`-in-`td` closing: fidelity and safety are the same
        # claim here — what the sanitizer sees is what the frame will show.
        html = body("<table><tr><td>a<p>x</td></tr></table>")
        self.assertIn("<tbody>", html)
        self.assertIn("<p>x</p>", html)


@unittest.skipUnless(HAVE_NH3, NEEDS_NH3)
class Stylesheets(unittest.TestCase):
    def test_the_head_stylesheet_survives_and_the_overlays_do_not(self):
        css = ('body{background:#f2f2f2}'
               '.h1{font-family:Arial;font-size:22px;color:#111}'
               '@media only screen and (max-width:480px)'
               '{.stack{display:block !important;width:100% !important}}'
               '@supports (display:grid){.g{display:grid;gap:8px}}'
               '@import url("http://evil/track.css");'
               '@font-face{font-family:X;src:url(http://evil/x.woff)}'
               '@keyframes spin{from{transform:rotate(0)}}'
               '.hide{position:fixed;top:0;left:0;z-index:99;background:#fff}'
               '.phish{background:url(http://evil/overlay.png) no-repeat}')
        html = body(f"<html><head><style>{css}</style></head>"
                    f"<body><p class=h1>Hi</p></body></html>")
        sheets = STYLE_SPANS.findall(html)
        self.assertTrue(sheets, "the mail's stylesheet disappeared entirely")
        sheet = "".join(sheets)
        self.assertIn(".h1{font-family:Arial", sheet)          # the mail's own type
        self.assertIn("@media only screen and (max-width:480px)", sheet)
        self.assertIn("@supports (display:grid)", sheet)
        self.assertIn("display:block !important", sheet)       # Gmail honours these
        self.assertNotIn("@import", sheet)
        self.assertNotIn("font-face", sheet)
        self.assertNotIn("keyframes", sheet)
        self.assertNotIn("position", sheet)
        self.assertNotIn("z-index", sheet)
        self.assertNotIn("url(", sheet)
        self.assertNotIn("evil", sheet)
        self.assertIn("background:#fff", sheet)                # kept, minus the overlay

    def test_css_text_can_never_end_its_own_element(self):
        """The invariant that makes keeping `<style>` defensible: no `<` survives
        inside a stylesheet, so CSS cannot close its element and become markup.
        """
        cases = [
            ('a{content:"</style><img src=x onerror=alert(1)>"}', True),
            ("a{color:red}</style><script>alert(1)</script>", True),
            ('a{background:url("evil")}</style><b>x</b>', True),
            ("@import'</style><style>.a{position:fixed}</style>", True),
            # An *unclosed* `<iframe>` swallows the rest of the document in a
            # browser too (everything after it is the element's fallback text),
            # so the sanitizer dropping it is fidelity, not a lost paragraph —
            # asserted here so the choice stays a known one.
            ('a{color:red}</style><iframe src="http://evil">', False),
        ]
        for payload, keeps_text in cases:
            html = body("<style>" + payload + "</style><p>text</p>")
            for sheet in STYLE_SPANS.findall(html):
                self.assertNotIn("<", sheet, f"escaped the stylesheet: {html}")
            low = html.lower()
            self.assertNotIn("onerror", low)
            self.assertNotIn("<script", low)
            self.assertNotIn("<iframe", low)
            self.assertNotIn("position", low)
            self.assertNotIn("evil", low)
            if keeps_text:
                self.assertIn("text", html, f"lost the mail for nothing: {html}")

    def test_broken_stylesheet_is_left_out_and_said_so(self):
        out = render("<style>.a{color:red</style><p>body</p>")   # rule never closed
        self.assertIn("body", out["html"])
        self.assertNotIn("<style", out["html"])
        self.assertTrue(any("stylesheet" in n for n in out["notes"]),
                        "a silently half-dropped stylesheet is a wrong-looking mail")

    def test_comments_inside_css_are_gone(self):
        html = body('<style>/*]]>*/*--!><a>.a{color:red}/*<img src=x onerror=1>*/'
                    '.b{color:blue}</style>')
        self.assertNotIn("/*", html)
        self.assertNotIn("onerror", html)
        self.assertIn(".b{color:blue}", html)

    def test_word_wrapping_its_stylesheet_in_html_comments_costs_nothing(self):
        """Outlook and Word write `<style><!-- … --></style>` in every document
        they produce. CSS ignores those two tokens, and so does a browser; here an
        unclaimed `-->` used to read as a rule with no block, so the reader was
        told part of the stylesheet had been left out when nothing had been — a
        note on half the mail in a normal mailbox.
        """
        out = render("<style><!--\n.a{color:red}"
                     "@page WordSection1{size:612pt 792pt}"
                     ".b{color:blue}\n--></style><p>body</p>")
        sheet = "".join(STYLE_SPANS.findall(out["html"]))
        self.assertIn(".a{color:red}", sheet)
        self.assertIn(".b{color:blue}", sheet)
        self.assertNotIn("page", sheet)                  # print rules, dropped
        self.assertFalse(any("stylesheet" in n for n in out["notes"]),
                         f"invented a note: {out['notes']}")

    def test_an_unreadable_at_rule_costs_its_own_block_and_no_more(self):
        """`@page` is known and dropped by name; an at-rule nobody knows (`@top-center`,
        a vendor one, whatever next) is still readable *as a block*, so only that
        block may go. Ending the scan there would silently delete the rest of the
        stylesheet, which is exactly the mail the reader cannot explain.
        """
        out = render("<style>.a{color:red}@-ms-viewport{width:device-width}"
                     "@top-center{content:\"x\"}.b{color:blue}</style><p>x</p>")
        sheet = "".join(STYLE_SPANS.findall(out["html"]))
        self.assertIn(".a{color:red}", sheet)
        self.assertIn(".b{color:blue}", sheet)
        self.assertNotIn("viewport", sheet)
        self.assertNotIn("top-center", sheet)


@unittest.skipUnless(HAVE_NH3, NEEDS_NH3)
class StyleAttributes(unittest.TestCase):
    def test_positioning_is_stripped_and_hiding_is_honoured(self):
        html = body('<div style="color:red;position:fixed;top:0;z-index:9;'
                    'background:url(http://evil/x.png)">x</div>'
                    '<p style="display:none">preview text</p>'
                    '<p style="visibility:hidden">also hidden</p>')
        # One dangerous declaration must not take the harmless ones with it:
        # Gmail keeps the colour, drops the overlay, and this matches it.
        self.assertIn("color:red", html)
        self.assertNotIn("position", html)
        self.assertNotIn("z-index", html)
        self.assertNotIn("evil", html)
        self.assertIn('style="display:none"', html)
        self.assertIn('style="visibility:hidden"', html)

    def test_a_style_attribute_cannot_carry_a_url_or_a_script(self):
        for payload in ('background:url(http://evil/t.png)',
                        'background:url(&quot;http://evil/t.png&quot;)',
                        'width:expression(alert(1))',
                        'background:url(javascript:alert(1))',
                        'behavior:url(#default#time2)',
                        '-moz-binding:url(http://evil/x.xml#y)'):
            html = body(f'<td style="{payload}">x</td>')
            self.assertNotIn("style=", html, f"kept a dangerous style: {html}")
            self.assertNotIn("evil", html)


@unittest.skipUnless(HAVE_NH3, NEEDS_NH3)
class Images(unittest.TestCase):
    REMOTE = '<img src="http://track.example/o.gif" width=1 height=1 alt="">'

    def test_remote_images_are_hidden_until_asked(self):
        off = render(self.REMOTE)
        self.assertEqual(off["images_blocked"], 1)
        self.assertEqual(off["images_shown"], 0)
        self.assertNotIn("track.example", off["html"])
        # The placeholder keeps the sender's own box: hidden images should not
        # collapse a table cell, which is what Gmail's hidden state looks like.
        self.assertIn("width=", off["html"])
        on = render(self.REMOTE, images_allowed=True)
        self.assertEqual(on["images_shown"], 1)
        self.assertIn('src="http://track.example/o.gif"', on["html"])

    def test_a_relative_url_cannot_ask_this_app_for_something(self):
        """A sandboxed frame has no same-origin permission, but it does inherit
        its host document's base URL: `/api/...` inside a mail is a request to
        the mailbox, and `//host/` is a request somewhere else.
        """
        html = body('<img src="/api/message/XYZ/attachment/secret.pdf">'
                    '<img src="//evil.example/x.png">'
                    '<a href="/api/status">look</a><a href="../secret">up</a>')
        self.assertNotIn("/api/", html)
        self.assertNotIn("//evil", html)
        self.assertNotIn("../", html)
        self.assertIn("look", html)               # the words stay readable

    def test_cid_becomes_a_local_attachment_and_a_missing_one_disappears(self):
        html = body('<img src="cid:logo@mailer" width=140 height=32 alt="Acme">'
                    '<img src="cid:not_in_this_mail@x">',
                    cid_map={"logo@mailr": "logo (1).png",
                             "logo@mailer": "logo (1).png"})
        self.assertIn(f'src="{MSG_URL}/logo%20%281%29.png"', html)
        self.assertIn('alt="Acme"', html)
        out = render('<img src="cid:gone@mailer">')
        self.assertNotIn("src=", out["html"])
        self.assertTrue(any("cid:" in n for n in out["notes"]))

    def test_an_inlined_image_is_judged_by_size(self):
        small = '<img src="data:image/png;base64,' + "A" * 500 + '">'
        huge = '<img src="data:image/png;base64,' + "A" * 400_000 + '">'
        self.assertIn("data:image/png", body(small))
        big = render(huge)
        self.assertNotIn("image/png", big["html"])    # the sender's payload is gone
        self.assertIn("image/gif", big["html"])       # the placeholder is in its place
        self.assertTrue(any("too big" in n for n in big["notes"]))

    def test_images_from_a_style_are_not_a_back_door(self):
        html = body('<td style="background-image:url(http://evil/pixel.gif)">x</td>',
                    images_allowed=True)
        self.assertNotIn("evil", html)


@unittest.skipUnless(HAVE_NH3, NEEDS_NH3)
class Links(unittest.TestCase):
    def test_links_leave_the_frame_and_cannot_take_it_with_them(self):
        html = body('<a href="http://e/x" target="_top">x</a>'
                    '<a href="http://e/y">y</a><a href="mailto:a@b.c">m</a>')
        self.assertEqual(html.count('target="_blank"'), 3)
        self.assertEqual(html.count('rel="noopener noreferrer"'), 3)
        self.assertNotIn("_top", html)
        self.assertIn('href="mailto:a@b.c"', html)

    def test_no_link_frames_the_pane(self):
        html = body('<a href="http://e/x"><iframe src="http://e/frame"></iframe></a>')
        self.assertNotIn("iframe", html.lower())


@unittest.skipUnless(HAVE_NH3, NEEDS_NH3)
class MailStillLooksLikeMail(unittest.TestCase):
    """Everything a real newsletter is built from, kept on purpose."""

    NEWSLETTER = (
        '<table width="600" border="0" cellpadding="0" cellspacing="0" '
        'align="center" bgcolor="#ffffff" style="width:600px"><tr>'
        '<td align="left" valign="top" bgcolor="#f4f4f4" nowrap="nowrap" '
        'width="200" style="padding:12px;font-family:Arial,sans-serif">'
        '<font face="Arial" size="2" color="#333333">Quarterly&nbsp;report'
        '</font><br><font color="#0066cc">Q3</font></td>'
        '<td align="right"><!--[if mso]><b>OUTLOOK</b><![endif]-->'
        '<span class="stack">see&nbsp;you&nbsp;there</span></td></tr></table>')

    def test_the_layout_markup_survives(self):
        html = body(self.NEWSLETTER)
        for needle in ('width="600"', 'bgcolor="#ffffff"', 'cellpadding="0"',
                       'cellspacing="0"', 'align="center"', 'valign="top"',
                       'nowrap="nowrap"', '<font face="Arial"', 'size="2"',
                       'color="#333333"', '&nbsp;', "<br>", 'style="padding:12px',
                       'font-family:Arial'):
            self.assertIn(needle, html, f"lost {needle}:\n{html}")
        self.assertNotIn("OUTLOOK", html)
        self.assertIn("see", html)

    def test_output_is_a_fragment_for_our_own_document(self):
        """The frame document, its CSP and its defaults are `frame_document`'s;
        a mail's `<html>`/`<head>`/`<body>` must not arrive to compete with them.
        """
        html = body('<html><head><style>p{color:red}</style><title>t</title></head>'
                    '<body bgcolor="red"><p>hi</p></body></html>')
        low = html.lower()
        for tag in ("<html", "<head", "<body", "<title", "</html"):
            self.assertNotIn(tag, low, f"a document inside our document: {html}")
        self.assertIn("color:red", low)               # its styles still apply

    def test_bidi_and_abbreviations_keep_their_markers(self):
        html = body('<div dir="rtl" lang="ar">مرحبا</div>'
                    '<abbr title="Quarterly business review">QBR</abbr>'
                    '<blockquote><p>quoted</p></blockquote>')
        self.assertIn('dir="rtl"', html)
        self.assertIn('lang="ar"', html)
        self.assertIn('title="Quarterly business review"', html)
        self.assertIn("<blockquote>", html)


@unittest.skipUnless(HAVE_NH3, NEEDS_NH3)
class Guards(unittest.TestCase):
    def test_a_deep_document_is_refused_before_it_costs_seconds(self):
        html = "<div>" * 100_000 + "x" + "</div>" * 100_000
        started = time.time()
        with self.assertRaises(mailhtml.MailHtmlTooDeep) as raised:
            mailhtml.render_email_html(html)
        took = time.time() - started
        self.assertLess(took, 1.0, f"the guard itself took {took:.1f}s")
        self.assertIn("deep", str(raised.exception))

    def test_a_big_payload_is_refused_with_its_size(self):
        html = "<p>" + "x" * (mailhtml.MAX_BYTES + 10) + "</p>"
        with self.assertRaises(mailhtml.MailHtmlTooLarge) as raised:
            mailhtml.render_email_html(html)
        self.assertIn("characters", str(raised.exception))

    def test_a_long_but_shallow_mail_is_rendered(self):
        html = "<p>line</p>" * 40_000
        started = time.time()
        out = mailhtml.render_email_html(html)
        self.assertLess(time.time() - started, 5.0)
        self.assertIn("<p>line</p>", out["html"])

    def test_a_mail_that_uses_many_void_tags_is_not_considered_nested(self):
        html = "<br>" * 5_000 + "<p>end</p>"
        self.assertIn("end", mailhtml.render_email_html(html)["html"])

    def test_the_two_tag_lists_must_stay_disjoint(self):
        """`ammonia` *panics* (a Rust panic, not a catchable Python exception)
        when one tag is both allowed and a clean-content tag. `style` is the tag
        that tempts this mistake, and it is exactly why it lives in `EMAIL_TAGS`
        with the stylesheet narrowing on the other side.
        """
        self.assertTrue(mailhtml.EMAIL_TAGS.isdisjoint(mailhtml.DROP_CONTENT_TAGS),
                        mailhtml.EMAIL_TAGS & mailhtml.DROP_CONTENT_TAGS)

    def test_an_empty_part_is_not_an_error(self):
        for blank in ("", "   ", "\n\t"):
            out = mailhtml.render_email_html(blank)
            self.assertEqual(out["html"], "")
            self.assertTrue(out["notes"])


@unittest.skipUnless(HAVE_NH3, NEEDS_NH3)
class FrameDocument(unittest.TestCase):
    def test_the_policy_is_written_before_the_mail_arrives(self):
        """`frame_document` is a template, not a second sanitizer: it takes the
        fragment `render_email_html` produced, and the point of writing the CSP
        before splicing it in is that a sender never negotiates with it.
        """
        payload = '</style><img src=x onerror=alert(1)><script>alert(2)</script>'
        styled = '<style>a{content:"' + payload + '"}</style><p>mail body</p>'
        doc = mailhtml.frame_document(render(styled)["html"])
        self.assertLess(doc.index("Content-Security-Policy"), doc.index("mail body"))
        self.assertNotIn("onerror", doc)
        self.assertNotIn("<script", doc.lower())
        self.assertIn("<style>", doc)                 # our defaults, not the mail's

    def test_the_frame_fetches_nothing_until_images_are_asked_for(self):
        blocked = mailhtml.frame_document("<p>x</p>", images_allowed=False)
        allowed = mailhtml.frame_document("<p>x</p>", images_allowed=True)
        csp = re.search(r'Content-Security-Policy" content="([^"]+)"', blocked)
        self.assertTrue(csp, blocked)
        self.assertIn("default-src 'none'", csp.group(1))
        self.assertIn("img-src 'none'", csp.group(1))
        self.assertIn("form-action 'none'", csp.group(1))
        self.assertIn("img-src http: https: data:",
                      re.search(r'Content-Security-Policy" content="([^"]+)"',
                                allowed).group(1))

    def test_a_full_render_then_wrap_has_no_escape_hatch(self):
        doc = mailhtml.frame_document(render(
            '<style>a{content:"</style><img src=x onerror=alert(1)>"}</style>'
            '<script>alert(1)</script><p>mail</p>')["html"])
        self.assertIn("mail", doc)
        self.assertNotIn("onerror", doc)
        self.assertLess(doc.count("<style"), 3)       # ours plus at most the mail's


if __name__ == "__main__":
    unittest.main(verbosity=2)
