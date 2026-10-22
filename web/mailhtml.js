'use strict';

/* One message's HTML, shown without ever letting it touch this page.
 *
 * Every other file here builds nodes with `el()` and writes text with
 * `textContent`, because mail text is written by strangers. A mail's HTML is the
 * one thing that trick does not cover: the markup *is* the thing you came to
 * read, and there is no way to show it as text and keep the receipt looking like
 * a receipt. So this file is the only place in the app that handles it, and what
 * it does with it is to not handle it. The server sends the body back already
 * sanitized and wrapped in a document of its own; all this does is put that
 * document in a frame with no permission to run anything.
 *
 * Two things make that enough, and they are both load-bearing.
 *
 * `srcdoc` is *assigned*, never parsed into this page's tree. There is no
 * HTML string being handed to a parser that belongs to this document, so there
 * is no point at which a mail's markup is temporarily this page's own — the
 * thing that happens the moment markup is written *into* a live document, and
 * what the test that scans this file for those four words exists to prevent.
 * (The word itself is spelled nowhere in here, in prose or in code, so that a
 * grep for it across `web/` is a complete answer.)
 *
 * The sandbox grants neither `allow-scripts` nor `allow-same-origin`. A frame
 * without an origin of its own cannot call `/api/*`, cannot read this page,
 * cannot follow a redirect back to the app as itself, and cannot execute the
 * script its own content policy already refused to fetch. That is the same shape
 * Gmail chose: the mail is rendered *somewhere else*, not merely escaped in
 * place — escaping is a claim about a string, and this is a fact about a
 * document.
 *
 * `allow-popups` is the one permission left on, deliberately: a link in a mail
 * is meant to be followed by leaving, and Gmail opens those in a new tab too.
 * `allow-popups-to-escape-sandbox` is what keeps that honest — the new tab is
 * the *target's* page with no sandbox at all rather than a frame that could
 * script its opener, so the mail never gains a window onto this one.
 */
(function () {
  /* Exactly one string, carried here so the attribute a reviewer has to check
     is the attribute the code sets. Losing `allow-popups-to-escape-sandbox`, or
     ever adding `allow-scripts`, is the whole bug this file is guarding. */
  var SANDBOX = 'allow-popups allow-popups-to-escape-sandbox';

  /* The last few bodies, so flicking between the two views is not a round trip
     each way. Keyed by message and by the image choice, because those are the
     only two things an answer depends on: the cache on the server holds the
     *fetched* HTML, and this one holds the rendered document. */
  var shown = new Map();
  var KEEP = 8;

  function load(id, images) {
    var want = !!images;
    var k = id + (want ? '#1' : '#0');
    if (shown.has(k)) return Promise.resolve(shown.get(k));
    return fetch('/api/message/' + encodeURIComponent(id) + '/html'
                 + (want ? '?images=1' : ''))
      .then(function (r) {
        return r.json().catch(function () { return null; }).then(function (data) {
          if (!r.ok || !data || !data.html) {
            var err = new Error((data && data.error)
                                || (r.status + ' ' + r.statusText));
            /* `reason` is the server's word for *why* there is nothing to show:
               "this mail has no HTML part" is a fact about the message, so the
               switch may be disabled; anything else is a condition worth trying
               again. Recognizing the sentence instead would break the day it is
               reworded, which is why the answer carries the reason as a field. */
            err.reason = (data && data.reason) || '';
            throw err;
          }
          if (shown.size >= KEEP) shown.delete(shown.keys().next().value);
          shown.set(k, data);
          return data;
        });
      });
  }

  /* The frame, and the one assignment of mail-authored bytes anywhere in this
     app. `sandbox` goes on *before* `srcdoc`: an iframe takes its permissions
     when its document is created, so setting them after would mean the first
     parse happened with the defaults. */
  function frame(data) {
    var f = document.createElement('iframe');
    f.className = 'mail-frame';
    f.setAttribute('title', "the message's own HTML, with scripts disabled");
    f.setAttribute('sandbox', SANDBOX);
    /* A mail's images are other people's servers counting a load. The address of
       this page is nobody's business while doing it; the pixel still arrives. */
    f.setAttribute('referrerpolicy', 'no-referrer');
    f.srcdoc = data.html;
    return f;
  }

  window.MailHtml = { load: load, frame: frame, SANDBOX: SANDBOX };
}());
