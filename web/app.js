'use strict';

/* vectormailbox front-end.
 *
 * One rule runs through this file: **mail text is untrusted input.** A subject
 * is written by a stranger, so nothing that came out of an email is ever
 * assigned as markup or built as an HTML string — every node goes through
 * `el()`, which sets text with `textContent`. The highlighter splits the string
 * and appends <mark> nodes for the same reason: the obvious implementation
 * (wrap the matches in tags, then assign the whole thing as markup) is an XSS
 * hole in a mail reader, and it is the obvious way to write it.
 *
 * The second rule is that the last answer to arrive must be the answer that
 * belongs to the current query. Searches are debounced, and each gets a sequence
 * number; a slow hybrid query that comes back after you already typed something
 * else is discarded rather than painted over the newer results.
 */

/* ------------------------------------------------------------------ dom --- */

function el(tag, props, ...kids) {
  const n = document.createElement(tag);
  if (props) {
    for (const k in props) {
      const v = props[k];
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') n.className = v;
      else if (k === 'text') n.textContent = v;            // text only, never markup
      else if (k === 'dataset') Object.assign(n.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? '' : String(v));
    }
  }
  for (const kid of kids.flat(3)) {
    if (kid === null || kid === undefined || kid === false) continue;
    n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
}

const NS = 'http://www.w3.org/2000/svg';
function icon(paths, opts) {
  const s = document.createElementNS(NS, 'svg');
  s.setAttribute('viewBox', (opts && opts.view) || '0 0 20 20');
  const size = String((opts && opts.size) || 14);
  s.setAttribute('width', size);
  s.setAttribute('height', size);
  for (const p of paths) {
    const n = document.createElementNS(NS, 'path');
    n.setAttribute('d', p);
    n.setAttribute('class', 'ui-stroke');
    s.append(n);
  }
  return s;
}
const X_SVG = () => icon(['M5 5l10 10M15 5L5 15'], { size: 11 });

const $ = (id) => document.getElementById(id);
const NF = new Intl.NumberFormat('en-US');
const fmt = (n) => (typeof n === 'number' ? NF.format(n) : String(n ?? ''));

/* ---------------------------------------------------------------- state --- */

const state = {
  view: 'search',            // search | inbox | files
  q: '',
  image: null,               // { name, type, data } — a data URL, or null
  mode: 'hybrid',
  top: 10,
  knobs: {},                 // extra ranking overrides
  tiers: [],
  explain: false,
  chips: null,               // object => these filters are authoritative
  chipsText: '',             // the text to search alongside them
  items: [],                 // result or browse rows, in display order
  sel: -1,
  cursor: '',
  total: 0,
  sender: '',
  seq: 0,
  status: {},
  openId: '',
  lastMeta: null,
  /* Whether *this reader* asked to see a mail's own images, for this session.
     A per-mail choice would mean asking again in the next newsletter, which is
     how those prompts train people to click through anything. */
  mailImages: false,
};

/* ----------------------------------------------------------------- api --- */

async function api(path, opts) {
  const r = await fetch(path, Object.assign({
    headers: { 'Content-Type': 'application/json' },
  }, opts || {}));
  let data = null;
  try { data = await r.json(); } catch (e) { data = null; }
  if (!r.ok) {
    throw new Error((data && data.error) || `${r.status} ${r.statusText}`);
  }
  return data || {};
}

/* -------------------------------------------------------------- dates --- */

function asDate(s) {
  if (!s) return null;
  const d = new Date(s);
  return isNaN(d.getTime()) ? null : d;
}
function dayKey(d) { return d ? `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}` : ''; }
function dayLabel(d) {
  if (!d) return 'undated';
  const now = new Date();
  const diff = Math.round((new Date(now.getFullYear(), now.getMonth(), now.getDate())
    - new Date(d.getFullYear(), d.getMonth(), d.getDate())) / 86400000);
  if (diff === 0) return 'today';
  if (diff === 1) return 'yesterday';
  const opts = { weekday: 'short', day: 'numeric', month: 'short' };
  if (d.getFullYear() !== now.getFullYear()) opts.year = 'numeric';
  return new Intl.DateTimeFormat(undefined, opts).format(d);
}
function whenLabel(d) {
  if (!d) return '';
  const now = new Date();
  const sameYear = d.getFullYear() === now.getFullYear();
  const opts = { day: '2-digit', month: 'short' };
  if (!sameYear) opts.year = '2-digit';
  let s = new Intl.DateTimeFormat('en-GB', opts).format(d);
  if (dayKey(d) === dayKey(now)) {
    s += ' ' + new Intl.DateTimeFormat('en-GB',
      { hour: '2-digit', minute: '2-digit', hour12: false }).format(d);
  }
  return s;
}

/* --------------------------------------------------------- highlighting --- */

/* Words worth lighting: short ones paint half the page. */
function queryTokens(q) {
  if (!q) return [];
  const seen = new Set();
  for (const raw of String(q).toLowerCase().split(/[^0-9a-zà-ÿ_]+/i)) {
    if (raw.length >= 4) seen.add(raw);
  }
  const out = [...seen].slice(0, 12);
  // A term of six letters or more also lights its inflections: searching
  // "facturen" should light "facturen" and "facturenering", and BM25's prefix
  // rule on the server already matched them.
  for (const t of out.slice()) {
    if (t.length >= 7) seen.add(t.slice(0, t.length - 2));
  }
  return [...seen].sort((a, b) => b.length - a.length);
}

function escRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

function highlighted(text, tokens) {
  const frag = document.createDocumentFragment();
  if (!tokens || !tokens.length || !text) {
    frag.append(document.createTextNode(text || ''));
    return frag;
  }
  const re = new RegExp('(' + tokens.map(escRe).join('|') + ')', 'gi');
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) frag.append(document.createTextNode(text.slice(last, m.index)));
    frag.append(el('mark', { text: m[0] }));
    last = m.index + m[0].length;
    if (m[0].length === 0) re.lastIndex++;
  }
  if (last < text.length) frag.append(document.createTextNode(text.slice(last)));
  return frag;
}

/* ------------------------------------------------------------- chrome --- */

function toast(msg, ms) {
  const t = $('toast');
  t.textContent = '';
  t.append(el('div', { text: msg }));
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, ms || 5200);
}

function setDot(cls, text, title) {
  const dot = $('dot');
  dot.className = 'dot' + (cls ? ' ' + cls : '');
  $('pilltext').textContent = text;
  if (title) $('statuspill').title = title;
}

function renderStatus() {
  const s = state.status;
  if (s.loading) { setDot('busy', 'warming index', 'loading the index into memory'); return; }
  if (s.error) { setDot('warn', 'index unavailable', s.error); return; }
  const bits = [];
  if (s.messages) bits.push(`${fmt(s.messages)} mail`);
  if (s.last_indexed_date) bits.push(String(s.last_indexed_date).slice(0, 10));
  if (s.endpoint_ok === false && s.endpoint_configured) bits.push('endpoint down');
  let cls = 'ok';
  if (s.endpoint_ok === false && s.mode !== 'lexical') cls = 'warn';
  const tip = s.endpoint_ok === false
    ? ('embedding endpoint: ' + (s.endpoint_error || 'down'))
    : `${fmt(s.chunks)} chunks · ${fmt(s.vectors)} vectors · dim ${s.dim} · ${s.db_path}`;
  setDot(cls, bits.join(' · ') || 'ready', tip + droppedNote(s));
}

// The index mirrors the folders it indexes: mail that left them — deleted in
// Gmail, or archived out of INBOX — leaves the index at the next refresh. This
// interface has no delete button and never gets one, so the tooltip is where it
// says what it dropped and when, instead of the mail quietly being gone.
function droppedNote(s) {
  const n = Number(s.removed_total || 0);
  if (!n) return '';
  // last_removal_at, not last_reconcile_at: that one moves on every check, and a
  // check that removed nothing did not drop anything. And rendered by whenLabel,
  // the label every other date here uses: the stored stamp is UTC, and slicing
  // it would print "14:52" where the page's own clock says 16:52.
  const at = whenLabel(asDate(s.last_removal_at));
  return ` · ${fmt(n)} message${n === 1 ? '' : 's'} Gmail no longer has`
    + `${at ? `, last dropped ${at}` : ''}`;
}

// What a finished refresh is worth saying. A run can have removed as well as
// added, and the removal is the part nobody expects, so it goes in the sentence
// rather than only in the server log.
function refreshDone(r) {
  const s = state.status || {};
  const rec = s.last_reconcile || {};
  const at = Date.parse(s.last_reconcile_at || '');
  // Only count this run's reconcile: an older one is last week's news. Note the
  // two clocks: Date.parse is in milliseconds, started_at is a Python
  // time.time() in seconds — comparing them raw says "fresh" forever.
  const fresh = !isNaN(at) && (!r || !r.started_at
    || at / 1000 >= Number(r.started_at) - 5);
  const gone = fresh ? Number(rec.removed || 0) : 0;
  const back = fresh ? Number(rec.restored || 0) : 0;
  const plural = (n, word) => `${fmt(n)} ${word}${n === 1 ? '' : 's'}`;
  if (gone && back) {
    return `index updated — ${plural(gone, 'message')} Gmail no longer has were `
      + `dropped from the index, ${plural(back, 'one')} that came back re-indexed`;
  }
  if (gone) {
    return `index updated — ${plural(gone, 'message')} Gmail no longer has `
      + `${gone === 1 ? 'was' : 'were'} dropped from the index`;
  }
  if (back) {
    return `index updated — ${plural(back, 'message')} that came back to Gmail `
      + `${back === 1 ? 'was' : 'were'} re-indexed`;
  }
  return 'index updated — the search engine has re-opened the mailbox';
}

async function pollStatus() {
  try {
    state.status = await api('/api/status');
    // Until the human touches a knob, the panel mirrors the config the server
    // actually loaded — showing 0.30 while the config says 0.45 would lie.
    if (!state.touched) {
      state.top = state.status.top || state.top;
      $('t-top').value = state.top;
      if (state.status.min_relevance != null) {
        $('t-min').value = Number(state.status.min_relevance).toFixed(2);
      }
      $('t-collapse').checked = !!state.status.collapse;
      $('t-tiernorm').checked = state.status.tier_norm !== false;
    }
    renderStatus();
  } catch (e) {
    setDot('warn', 'server down', String(e));
  }
}

async function loadFacets() {
  try {
    const f = await api('/api/facets');
    const t = $('totals');
    t.textContent = '';
    const tot = f.totals || {};
    const rows = [
      ['messages', fmt(tot.messages)],
      ['threads', fmt(tot.threads)],
      ['with files', fmt(tot.with_attachments)],
      ['oldest', (tot.oldest || '').slice(0, 7)],
      ['newest', (tot.newest || '').slice(0, 7)],
    ];
    for (const [k, v] of rows) {
      if (!v || v === '0') continue;
      t.append(el('dt', { text: k }), el('dd', { text: v }));
    }
    const ul = $('senders');
    ul.textContent = '';
    const filter = $('sender-filter').value.trim().toLowerCase();
    for (const s of (f.senders || [])) {
      if (filter && !String(s.sender).toLowerCase().includes(filter)) continue;
      ul.append(el('li', {},
        el('button', {
          class: 'sender' + (state.sender === s.sender ? ' is-on' : ''),
          title: s.sender,
          onclick: () => pickSender(state.sender === s.sender ? '' : s.sender),
        },
          el('span', { class: 's-name', text: s.sender || '(no sender)' }),
          el('span', { class: 's-n', text: fmt(s.count) }))));
    }
  } catch (e) { /* the rail is decoration; the list is the product */ }
}

/* --------------------------------------------------------------- banners -- */

function banner(kind, htmlBits) {
  const b = el('div', { class: 'banner' + (kind ? ' ' + kind : '') });
  for (const bit of htmlBits) {
    if (typeof bit === 'string') b.append(document.createTextNode(bit));
    else b.append(bit);
  }
  return b;
}

function renderBanners(meta, degraded) {
  const box = $('banners');
  box.textContent = '';
  const s = state.status;
  if (s.error) {
    box.append(banner('warn', ['the index cannot be read: ',
      el('b', { text: s.error }),
      ' — run ', el('code', { text: 'python mailbox.py index --full' }), ' first.']));
    return;
  }
  if (degraded) box.append(banner('warn', [degraded, '.']));
  if (meta && meta.image) {
    // The picture is the part of the query the list cannot show, so it is named
    // here. With only a picture there is nothing further to say — the weights in
    // the sub-line already read `dense`. With both, how they combine is not
    // obvious from the list, so that one sentence is worth its space.
    const bits = ['searched by picture: ', el('b', { text: meta.image })];
    if (!meta.image_only) {
      bits.push(' — a chunk is scored on its better side, the picture or the '
        + 'words, and the words run through bm25 as usual.');
    }
    box.append(banner('', bits));
  }
  if (meta && meta.abstained) {
    if (meta.note) {
      // The filters alone were decisive: nothing can match them, so no search
      // ran. Saying "best 0.000 is under the floor" here would describe a
      // ranking that never happened, and would hide the one useful fact — that
      // a chip is what emptied the list.
      box.append(banner('warn', [meta.note, ' — remove a chip to widen it again.']));
    } else {
      box.append(banner('warn', [
        'nothing above the relevance floor — best ',
        el('b', { text: Number(meta.relevance || 0).toFixed(3) }),
        ' is under ', el('b', { text: Number(meta.min_relevance || 0).toFixed(2) }),
        ': the list below is the closest mail, not an answer.']));
    }
  }
  if (meta && meta.filters && state.chips === null) {
    box.append(banner('', ['read from your sentence: ', el('b', { text: meta.filters })]));
  }
}

/* ---------------------------------------------------------------- list --- */

/* The row's picture: the first attachment of this message that has a preview
   next to the index. It costs one query for the whole page (`first_for_messages`
   on the server) and a 30 px file from the same origin, which is why a list of
   mail can show its pictures while it is still being drawn. `loading="lazy"`
   keeps the ones below the fold from asking at all. */
function rowThumb(item) {
  if (!item.thumb) return null;
  return el('img', {
    class: 'r-thumb', src: mediaUrl(item.message_id, item.thumb), alt: '',
    loading: 'lazy', title: 'preview of ' + item.thumb,
  });
}

function legRow(denseRank, lexRank) {
  /* Fill length is rank, not score: rank 1 fills the hairline, rank 20 is a
     tick. `dense_rank`/`bm25_rank` are what the fusion actually consumed. */
  const fill = (r) => (r ? Math.max(6, Math.round(100 / (1 + (r - 1) / 3.2))) : 0);
  const box = el('div', { class: 'legs' });
  const d = el('i'); const l = el('i');
  box.append(
    el('div', { class: 'leg dense', title: denseRank ? `vector rank #${denseRank}` : 'not found by the vector leg' }, d),
    el('div', { class: 'leg lex', title: lexRank ? `keyword rank #${lexRank}` : 'not found by the keyword leg' }, l));
  requestAnimationFrame(() => { d.style.width = fill(denseRank) + '%'; l.style.width = fill(lexRank) + '%'; });
  return box;
}

function resultRow(h, i) {
  const toks = queryTokens(state.q);
  const d = asDate(h.date);
  const snippet = h.snippet || (h.text || '').replace(/\s+/g, ' ').trim().slice(0, 190);
  const tags = [];
  if (h.tier) tags.push(el('span', { class: 'tag tier', title: 'the chunk that matched', text: h.tier }));
  if (h.also_in_thread) tags.push(el('span', { class: 'tag thread', title: 'other messages of this conversation were folded into this result', text: `+${h.also_in_thread} in thread` }));
  const exts = (h.label && /attachment:/.test(h.label)) ? 'attachment' : '';
  if (exts) tags.push(el('span', { class: 'tag att', title: h.label, text: 'in attachment' }));
  if (state.explain) {
    tags.push(el('span', {
      class: 'tag',
      title: 'explain',
      text: `fused ${(h.fused || 0).toFixed(3)} · prior ${(h.prior || 1).toFixed(2)}`
        + (h.rerank != null ? ` · rerank ${h.rerank.toFixed(3)}` : ''),
    }));
  }
  const row = el('button', {
    class: 'row' + (h.thumb ? ' has-thumb' : '')
      + (i === state.sel ? ' is-sel' : '') + (h.score < 0.3 ? ' below' : ''),
    dataset: { id: h.message_id },
    onclick: () => { select(i); open(h.message_id); },
  },
    rowThumb(h),
    el('div', {},
      el('div', { class: 'r-line1' },
        el('span', { class: 'r-from', text: h.sender || '(no sender)' }),
        el('span', { class: 'r-when', text: whenLabel(d) })),
      el('div', { class: 'r-subject' }, highlighted(h.subject || '(no subject)', toks)),
      snippet ? el('div', { class: 'r-snippet' }, highlighted(snippet, toks)) : null,
      tags.length ? el('div', { class: 'r-tags' }, tags) : null),
    el('div', { class: 'r-metric' },
      legRow(h.dense_rank, h.bm25_rank),
      el('span', {
        class: 'r-score',
        title: 'fused score (1.0 = ranked first by both legs)',
        text: h.score.toFixed(3),
      })));
  return row;
}

function browseRow(m, i) {
  const d = asDate(m.date);
  const toks = queryTokens(state.textF || '');
  const exts = (m.attachment_exts || '').split(',').filter((x) => x && x !== 'none');
  const tags = exts.length
    ? [el('span', { class: 'tag att', title: exts.join(', '), text: exts.slice(0, 4).join(' · ') })]
    : [];
  if (m.thread_len > 1) tags.push(el('span', { class: 'tag thread', text: `${m.thread_len} in thread` }));
  return el('button', {
    class: 'row' + (m.thumb ? ' has-thumb' : '') + (i === state.sel ? ' is-sel' : ''),
    dataset: { id: m.message_id },
    onclick: () => { select(i); open(m.message_id); },
  },
    rowThumb(m),
    el('div', {},
      el('div', { class: 'r-line1' },
        el('span', { class: 'r-from', text: m.sender || '(no sender)' }),
        el('span', { class: 'r-when', text: whenLabel(d) })),
      el('div', { class: 'r-subject' }, highlighted(m.subject || '(no subject)', toks)),
      m.preview ? el('div', { class: 'r-snippet' }, highlighted(m.preview, toks)) : null,
      tags.length ? el('div', { class: 'r-tags' }, tags) : null),
    el('div', { class: 'r-metric' },
      el('span', { class: 'r-score', text: '' })));
}

function renderRows(kind) {
  const box = $('rows');
  box.textContent = '';
  if (!state.items.length) return;
  let lastDay = null;
  state.items.forEach((it, i) => {
    const d = asDate(it.date);
    const key = dayKey(d);
    if (key !== lastDay) {
      lastDay = key;
      box.append(el('div', { class: 'day', text: dayLabel(d) }));
    }
    box.append(kind === 'search' ? resultRow(it, i) : browseRow(it, i));
  });
}

function skeleton(n) {
  const box = $('rows');
  box.textContent = '';
  for (let i = 0; i < (n || 6); i++) {
    box.append(el('div', { class: 'skel' }, el('i'), el('i'), el('i')));
  }
}

function noteEmpty(html) {
  const box = $('rows');
  box.textContent = '';
  box.append(el('div', { class: 'empty-note' }, html));
}

function listSub(bits) {
  const sub = $('list-sub');
  sub.textContent = '';
  sub.append(highlightJoin(bits));
}
function highlightJoin(bits) {
  const frag = document.createDocumentFragment();
  for (const b of bits) {
    if (b === null || b === '') continue;
    frag.append(typeof b === 'string' ? document.createTextNode(b) : el('b', { text: String(b) }));
  }
  return frag;
}

/* ------------------------------------------------------------- searching -- */

function chipSpec(chips) {
  const out = [];
  const push = (field, value, label, key) => out.push({ field, value, label, key });
  for (const f of ['senders', 'recipients', 'subjects']) {
    const word = { senders: 'from', recipients: 'to', subjects: 'subject' }[f];
    for (const v of (chips[f] || [])) push(f, v, word, `${f}:${v}`);
  }
  if (chips.since) push('since', chips.since, 'since');
  if (chips.until) push('until', chips.until, 'until');
  if (chips.has_attachment === true) push('has_attachment', true, 'with files');
  if (chips.has_attachment === false) push('has_attachment', false, 'no files');
  return out;
}

function renderChips() {
  const box = $('chips');
  box.textContent = '';
  if (!state.chips) return;
  for (const c of chipSpec(state.chips)) {
    box.append(el('span', { class: 'chip', title: 'applied as a filter before anything is scored' },
      el('b', { text: c.label }),
      c.value !== true && c.value !== false ? document.createTextNode(' ' + String(c.value)) : null,
      el('button', {
        class: 'chip-x', title: 'drop this filter and search again',
        'aria-label': 'drop filter',
        onclick: (ev) => { ev.stopPropagation(); dropChip(c); },
      }, X_SVG())));
  }
}

function dropChip(c) {
  const next = Object.assign({}, state.chips);
  if (['senders', 'recipients', 'subjects'].includes(c.field)) {
    next[c.field] = (next[c.field] || []).filter((v) => v !== c.value);
  } else if (c.field === 'has_attachment') {
    delete next.has_attachment;
  } else {
    delete next[c.field];
  }
  state.chips = next;                       // may be {} : searched as typed
  runSearch(true);
}

/* ------------------------------------------------------------ image query -- */

/* The server's cap (gui.MAX_IMAGE_BYTES), repeated here so an oversized file is
   dealt with before it is read into a base64 string, not after it has been sent.
   The bytes never round-trip: the thumbnail below is painted from the same data
   URL the server gets, so nothing a stranger mailed can be served back from this
   origin by way of a search box. */
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
/* The side `mailbox.resize_image_b64` reduces every image to on the way to the
   model. A phone photo is 8-25 MB and the model never sees more than this, so a
   file over the cap is not a bigger search — it is the same search wrapped in
   30 MB of JSON, refused for a reason the browser can fix. Hence `shrinkImage`
   instead of a message about it. */
const MAX_IMAGE_SIDE = 1024;
/* SVG is deliberately absent: it is a script that happens to draw something, and
   this one draws into a page that holds your whole mailbox. */
const IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif'];

const MB = (n) => (n / 1048576).toFixed(1) + ' MB';

/* Size to reduce to: the long side becomes `max`, never up-scaled, never below
   one pixel. Pure, and therefore testable without a browser. */
function fitSide(w, h, max) {
  const s = Math.min(1, max / Math.max(w, h, 1));
  return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
}

/* Bytes behind a data URL, without decoding 30 MB of base64 to find out. The
   padding is what a `length * 3/4` shortcut would count as payload: five bytes
   travel as eight characters. */
function dataBytes(url) {
  const b64 = url.slice(url.indexOf(',') + 1);
  const pad = b64.endsWith('==') ? 2 : (b64.endsWith('=') ? 1 : 0);
  return Math.max(0, Math.floor(b64.length * 3 / 4) - pad);
}

const readAsDataURL = (blob) => new Promise((res, rej) => {
  const fr = new FileReader();
  fr.onload = () => res(String(fr.result || ''));
  fr.onerror = () => rej(new Error('unreadable'));
  fr.readAsDataURL(blob);
});

/* Draw the file at the size the endpoint would have used anyway and hand back
   that smaller encoding. The type is kept wherever a canvas can write it, and a
   transparent PNG or WebP stays transparent: re-encoding a screenshot as JPEG
   would quietly paint black where its alpha was, and the model would search
   that. A GIF becomes a PNG, because no canvas writes GIF. */
async function shrinkImage(file) {
  const bmp = typeof createImageBitmap === 'function'
    ? await createImageBitmap(file)
    : await new Promise((res, rej) => {
        const url = URL.createObjectURL(file);
        const im = new Image();
        im.onload = () => { URL.revokeObjectURL(url); res(im); };
        im.onerror = () => { URL.revokeObjectURL(url); rej(new Error('undecodable')); };
        im.src = url;
      });
  const [w, h] = fitSide(bmp.width, bmp.height, MAX_IMAGE_SIDE);
  const cv = document.createElement('canvas');
  cv.width = w;
  cv.height = h;
  cv.getContext('2d').drawImage(bmp, 0, 0, w, h);
  if (bmp.close) bmp.close();        // a decoded 6000x4000 bitmap is ~96 MB
  const type = (file.type === 'image/png' || file.type === 'image/gif')
    ? 'image/png' : file.type;
  const blob = await new Promise((res) => cv.toBlob(res, type, 0.92));
  if (!blob) throw new Error('unencodable');
  return { data: await readAsDataURL(blob), bytes: blob.size, type, w, h };
}

async function setImage(file) {
  if (!file) return;
  if (!IMAGE_TYPES.includes(file.type)) {
    toast('‘' + (file.name || 'that file') + '’ is ' + (file.type || 'not a known type')
      + ' — search by picture takes PNG, JPEG, WebP or GIF.');
    return;
  }
  const name = file.name || 'pasted image';
  /* Whether this endpoint can read a picture at all is learned the first time
     one is tried. Said here, at the pick, where it costs the user something —
     not on the status pill, where it is a footnote under every search they run. */
  if (state.status.endpoint_images === false) {
    toast('this endpoint cannot read a picture — search by words instead');
    return;
  }
  let img;
  if (file.size > MAX_IMAGE_BYTES) {
    /* Paint something first, then work: decoding a 24 MB photo is the slow part,
       and a bar that changes only afterwards is a bar nobody believes ran. */
    state.image = { name, type: file.type, size: file.size, data: '', busy: true };
    renderImageChip();
    await new Promise((r) => requestAnimationFrame(r));
    try {
      const small = await shrinkImage(file);
      if (small.bytes > MAX_IMAGE_BYTES) {
        clearImage();
        toast('even at ' + small.w + '×' + small.h + ' px that is ' + MB(small.bytes)
          + ', over the ' + MB(MAX_IMAGE_BYTES) + ' a search image may be.');
        return;
      }
      img = { name, type: small.type, size: small.bytes, data: small.data,
              from: file.size, w: small.w, h: small.h };
    } catch (err) {
      clearImage();
      toast('your browser could not open ‘' + name + '’ as a picture.');
      return;
    }
  } else {
    try {
      img = { name, type: file.type, size: file.size, data: await readAsDataURL(file) };
    } catch (err) {
      toast('the image could not be read from disk.');
      return;
    }
  }
  state.image = img;
  renderImageChip();
  runSearch(true);
}

function clearImage() {
  if (!state.image) return;
  state.image = null;
  renderImageChip();
  runSearch(true);
}

function renderImageChip() {
  const im = state.image;
  const on = !!im;
  const btn = $('img-btn');
  btn.classList.toggle('is-on', on);
  btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  btn.title = on ? 'search by picture is on — remove it with the ✕ in the bar'
    : 'Search by picture — pick a file, paste a screenshot, or drop an image here';
  const chip = $('imgchip');
  chip.hidden = !on;
  chip.classList.toggle('is-busy', !!(im && im.busy));
  const thumb = $('img-thumb');
  if (!on) {
    thumb.removeAttribute('src');
    $('img-name').textContent = '';
    return;
  }
  if (im.busy) {
    $('img-name').textContent = 'reducing ' + MB(im.size) + ' photo…';
    thumb.removeAttribute('src');            // there is no thumbnail yet, so no circle
    thumb.title = '';
    return;
  }
  thumb.src = im.data;                       // a data: URL of an image type, never markup
  thumb.title = im.name + ' · ' + Math.round(dataBytes(im.data) / 1024) + ' KB sent'
    + (im.from ? ' · reduced from ' + MB(im.from) + ' to ' + im.w + '×' + im.h
               + ' px, which is the side the model sees' : '');
  $('img-name').textContent = im.name;
}

function searchBody() {
  const body = Object.assign({ q: state.q, mode: state.mode, top: state.top }, state.knobs);
  if (state.image) {
    body.image = state.image.data;
    body.image_name = state.image.name;
  }
  if (state.tiers.length) body.tiers = state.tiers.join(',');
  if (state.explain) body.explain = true;
  if (state.chips) {
    body.filters = state.chips;
    body.text = state.chipsText || state.q;
  }
  return body;
}

async function runSearch(keepChips) {
  const query = state.q.trim();
  if (!query && !state.image) { loadBrowse(true); return; }
  // An image-only search still needs a heading; the file's own name is the most
  // honest thing to call it, since nothing was typed.
  const title = query || (state.image ? state.image.name : '');
  if (!keepChips) { state.chips = null; state.chipsText = ''; }
  state.view = 'search';
  setActiveView('search');
  const seq = ++state.seq;
  state.sel = -1;
  skeleton(7);
  $('list-title').textContent = 'searching';
  listSub(['…']);
  $('more').hidden = true;
  try {
    const data = await api('/api/search', { method: 'POST', body: JSON.stringify(searchBody()) });
    if (seq !== state.seq) return;                 // a newer query already answered
    state.items = data.results || [];
    state.lastMeta = data.meta;
    const meta = data.meta || {};
    if (meta.filters_struct && !state.chips) {
      state.chips = meta.filters_struct;
      state.chipsText = meta.embed_text || query;
    }
    renderChips();
    renderBanners(meta, data.degraded);
    $('list-title').textContent = title;
    listSub(summarise(meta, data.took_ms));
    if (!state.items.length && !meta.abstained) {
      noteEmpty([
        el('b', { text: 'no message matched' }),
        meta.image
          // A picture that finds nothing has one honest sentence. Whether this
          // endpoint can read pictures at all was settled at the pick, and a
          // pointer at another part of the page is a second task, not an answer.
          ? ' — nothing in the index looked like this picture.'
          : [' — try ', el('b', { text: 'lexical' }),
             ' for exact words, or drop a filter chip.'],
      ]);
    } else {
      renderRows('search');
    }
    if (state.items.length) select(0, false);
  } catch (e) {
    if (seq !== state.seq) return;
    state.items = [];
    renderBanners(null, null);
    $('rows').textContent = '';
    $('rows').append(el('div', { class: 'empty-note' },
      el('b', { text: 'the search did not run' }), document.createTextNode(' — ' + String(e.message || e))));
    $('list-title').textContent = title;
    listSub(['error']);
  }
}

function summarise(meta, tookMs) {
  const legs = [];
  if (meta.w_dense > 0) legs.push(`dense ${meta.w_dense}`);
  if (meta.w_lex > 0) legs.push(`bm25 ${meta.w_lex}`);
  const bits = [
    `${fmt(state.items.length)} shown of `, fmt(meta.chunks_matched || 0), ' matched · ',
    // With nothing typed but a picture there is no keyword leg to fuse with, so
    // `hybrid` here would name a second engine that never ran - the weights
    // beside it already say dense-only.
    meta.image_only ? 'picture' : meta.mode, ' ', legs.join(' + ') || 'no legs', ' · ',
    tookMs != null ? `${(tookMs / 1000).toFixed(2)}s` : null, ' · ',
    fmt(meta.indexed_messages || 0), ' indexed',
  ];
  if (meta.image && !meta.image_only) bits.push(' · picture and words both scored');
  if (meta.collapsed) bits.push(' · ', meta.collapsed, ' folded into conversations');
  if (meta.tiers && meta.tiers.length) bits.push(' · tiers ', meta.tiers.join(','));
  return bits;
}

/* ------------------------------------------------------------- browsing --- */

async function loadBrowse(reset) {
  if (reset) { state.cursor = ''; state.items = []; state.sel = -1; }
  const params = new URLSearchParams();
  params.set('limit', '50');
  if (state.cursor) params.set('before', state.cursor);
  if (state.sender) params.set('sender', state.sender);
  if (state.view === 'files') params.set('has_attachment', 'true');
  if (state.textF) params.set('text', state.textF);
  const seq = ++state.seq;
  if (reset) skeleton(8);
  try {
    const data = await api('/api/messages?' + params.toString());
    if (seq !== state.seq) return;
    state.items = reset ? (data.items || []) : state.items.concat(data.items || []);
    state.cursor = data.next_cursor || '';
    state.total = data.total || 0;
    renderBanners(null, null);
    const who = state.sender ? state.sender : (state.view === 'files' ? 'with files' : 'recent mail');
    $('list-title').textContent = state.textF ? `${who} · ${state.textF}` : who;
    listSub([fmt(state.items.length), ' of ', fmt(state.total), ' messages',
      state.sender ? ' · sender filter on' : null]);
    if (!state.items.length) {
      // Which of the two empties this is decides what to do about it, and the
      // page already knows: a filter is either on the left or it is not.
      noteEmpty(state.sender || state.textF || state.view === 'files'
        ? [el('b', { text: 'nothing here' }),
           ' — no message matches the filter on the left.']
        : [el('b', { text: 'this index holds no messages' }), ' — run ',
           el('code', { text: 'python mailbox.py index' }), '.']);
    } else {
      renderRows('browse');
    }
    $('more').hidden = !state.cursor;
  } catch (e) {
    if (seq !== state.seq) return;
    noteEmpty([el('b', { text: 'the list did not load' }), document.createTextNode(' — ' + String(e.message || e))]);
  }
}

function pickSender(s) {
  state.sender = s;
  loadFacets();
  if (state.view === 'search' && state.q.trim()) return;   // sender chips are the filter there
  loadBrowse(true);
}

function setActiveView(v) {
  state.view = v;
  document.querySelectorAll('.view-btn').forEach((b) => {
    b.classList.toggle('is-active', b.dataset.view === v);
  });
}

/* --------------------------------------------------------------- reader --- */

function kv(k, v, cls) {
  return [el('dt', { text: k }), el('dd', { class: cls || '' }, String(v ?? ''))];
}

async function open(id) {
  state.openId = id;
  document.querySelectorAll('.row').forEach((r) => {
    r.classList.toggle('is-sel', r.dataset.id === id);
  });
  $('reader').removeAttribute('data-closed');
  const body = $('reader-body');
  body.hidden = false;
  body.textContent = '';
  body.append(el('div', { class: 'rb-inner' }, el('div', { class: 'rb-kicker', text: 'loading' })));
  $('reader-empty').hidden = true;
  try {
    const m = await api('/api/message/' + encodeURIComponent(id));
    if (state.openId !== id) return;
    renderMessage(m);
  } catch (e) {
    body.textContent = '';
    body.append(el('div', { class: 'rb-inner' },
      el('p', { class: 'rb-note warn', text: 'this message could not be read: ' + String(e.message || e) })));
  }
}

/* The file itself stays at Gmail: the index keeps words, never bytes. A picture
   is the exception worth making, because the indexer decoded and resized it once
   already — a small copy of it lives beside the index, so the first thing a click
   shows is on this machine before Gmail is asked for anything. The full file
   still comes from Gmail when it is opened, now as the one MIME part it is
   instead of as the whole message around it, and the browser is told to keep it. */
function attachmentUrl(mid, name) {
  return '/api/message/' + encodeURIComponent(mid)
    + '/attachment/' + encodeURIComponent(name);
}

function mediaUrl(mid, name) {
  return '/api/message/' + encodeURIComponent(mid)
    + '/media/' + encodeURIComponent(name);
}

const IMAGE_NAME = /\.(png|jpe?g|gif|webp)$/i;

/* Everything that is not a picture opens as itself: the server reads it back and
   answers with its own Content-Disposition, and the browser takes it from there.
   Fetching the blob here instead of following a plain link means a failure
   arrives as a sentence ("Gmail no longer answers for this message") and not as
   raw JSON opened in a new tab. */
async function openFile(mid, name) {
  try {
    const r = await fetch(attachmentUrl(mid, name));
    if (!r.ok) {
      let msg = r.status + ' ' + r.statusText;
      try { msg = (await r.json()).error || msg; } catch (e) { /* not json */ }
      throw new Error(msg);
    }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    try {
      const a = el('a', { href: url, download: name });
      document.body.append(a);
      a.click();
      a.remove();
    } finally {
      /* Revoke later: Firefox reads blob URLs asynchronously and a blob
         PDF opened straight from a revoked URL can come up blank. */
      setTimeout(() => URL.revokeObjectURL(url), 30000);
    }
  } catch (e) {
    toast('could not open ' + name + ': ' + String(e.message || e));
  }
}

function attChip(m, name, media) {
  const dot = name.lastIndexOf('.');
  const ext = dot > 0 ? name.slice(dot + 1) : '';
  const preview = media && media[name];
  // A picture opens over the mail it came from; a file opens as that file. The
  // chip says which of the two it is, because a control that sometimes downloads
  // and sometimes shows is the one people click by accident.
  const opens = Boolean(preview) || IMAGE_NAME.test(name);
  const chip = el('button', {
    class: 'att-chip' + (preview ? ' has-thumb' : ' is-open'),
    type: 'button',
    title: preview
      ? name + ' — opens the preview stored here'
      : name + ' — downloads the file from Gmail',
    onclick: (ev) => {
      const btn = ev.currentTarget;
      if (opens) { openLightbox(m, name, preview, btn); return; }
      btn.dataset.busy = '1';
      openFile(m.message_id, name).finally(() => { delete btn.dataset.busy; });
    },
  });
  if (preview) {
    chip.append(el('img', {
      class: 'att-thumb', src: mediaUrl(m.message_id, name), alt: '',
      loading: 'lazy', width: preview.width || 0, height: preview.height || 0,
    }));
  } else if (ext) {
    chip.append(el('span', { class: 'ext', text: ext }));
  }
  chip.append(el('span', { text: dot > 0 ? name.slice(0, dot) : name }));
  chip.append(opens
    ? icon(['M9 4.5A4.5 4.5 0 1 0 9 13.5 4.5 4.5 0 0 0 9 4.5zM12.6 12.6L16 16'],
      { size: 11 })
    : icon(['M10 4v8m0 0l-3-3m3 3l3-3M5 16h10'], { size: 11 }));
  return chip;
}

/* -- the preview box -------------------------------------------------------
   A picture opens over the mail it came from, with the two ways out that
   matter: the file itself, and Gmail. It opens on the local preview so the
   first frame is already there, and reads the full file from Gmail behind it.
   Nothing is prefetched anywhere else: a mailbox holds thousands of pictures,
   and the ones nobody opens should not cost a megabyte apiece.
   ------------------------------------------------------------------------ */
let lb = null;

function ensureLightbox() {
  if (lb) return lb;
  const nameEl = el('span', { class: 'lb-name', id: 'lb-name' });
  const actions = el('span', { class: 'lb-actions' });
  const stage = el('div', { class: 'lb-stage' });
  const note = el('p', { class: 'lb-note', hidden: true });
  const closeBtn = el('button', {
    class: 'lb-close', type: 'button', title: 'close (esc)', 'aria-label': 'close',
    onclick: () => closeLightbox(),
  }, X_SVG());
  const panel = el('div', {
    class: 'lb-panel', role: 'dialog', 'aria-modal': 'true',
    'aria-labelledby': 'lb-name',
  }, el('div', { class: 'lb-bar' }, nameEl, actions, closeBtn), stage, note);
  const root = el('div', {
    class: 'att-lightbox', hidden: true,
    // the backdrop closes; the picture never does
    onclick: (ev) => { if (ev.target === root) closeLightbox(); },
  }, panel);
  document.body.append(root);
  document.addEventListener('keydown', (ev) => {
    if (root.hidden) return;
    if (ev.key === 'Escape') { ev.preventDefault(); closeLightbox(); return; }
    if (ev.key !== 'Tab') return;
    // Three things to reach and no way out: a dialog that lets Tab wander back
    // into the mail behind it sends a keyboard user somewhere they did not ask
    // to be, and then leaves them looking for where they were.
    const items = [...root.querySelectorAll('a[href], button')];
    if (!items.length) return;
    const first = items[0];
    const last = items[items.length - 1];
    if (ev.shiftKey && document.activeElement === first) {
      ev.preventDefault();
      last.focus();
    } else if (!ev.shiftKey && document.activeElement === last) {
      ev.preventDefault();
      first.focus();
    }
  });
  lb = { root, nameEl, actions, stage, note, closeBtn, token: {}, back: null };
  return lb;
}

function openLightbox(m, name, preview, from) {
  const box = ensureLightbox();
  const token = {};
  box.token = token;                  // a close or a re-open voids the load below
  box.back = from || document.activeElement;
  box.nameEl.textContent = name;
  box.note.hidden = true;
  box.note.textContent = '';
  const g = gmailHref(m);
  box.actions.textContent = '';
  box.actions.append(
    el('a', {
      class: 'link-btn', href: attachmentUrl(m.message_id, name), download: name,
      title: 'the original file, from Gmail',
    }, 'download', icon(['M10 4v8m0 0l-3-3m3 3l3-3M5 16h10'], { size: 11 })),
    el('a', { class: 'link-btn', href: g.href, target: '_blank', rel: 'noreferrer',
      title: g.title }, g.label, icon(['M7 13l6-6M8 7h5v5'], { size: 11 })));

  const img = el('img', { class: 'lb-img', alt: name });
  const wait = el('span', {
    class: 'lb-wait',
    text: preview ? 'preview — reading the file from gmail'
                  : 'reading the file from gmail',
  });
  box.stage.textContent = '';
  box.stage.append(img, wait);
  if (preview) img.src = mediaUrl(m.message_id, name);
  else img.hidden = true;
  const full = new Image();
  full.onload = () => {
    if (box.token !== token) return;
    img.src = attachmentUrl(m.message_id, name);
    img.hidden = false;
    wait.remove();
  };
  full.onerror = () => {
    if (box.token !== token) return;
    wait.remove();
    // What is on screen, and the one move that can still get the original: the
    // download in the bar above hits the same URL again.
    box.note.hidden = false;
    box.note.textContent = preview
      ? 'showing the smaller copy saved when this mail was indexed — '
        + 'download asks Gmail again'
      : 'Gmail did not answer for this file — download asks it again';
  };
  full.src = attachmentUrl(m.message_id, name);
  box.root.hidden = false;
  box.closeBtn.focus();
}

function closeLightbox() {
  if (!lb || lb.root.hidden) return;
  lb.token = {};                      // the load in flight has nowhere to land
  lb.root.hidden = true;
  lb.stage.textContent = '';
  const back = lb.back;
  lb.back = null;
  // Home again: a box that eats the keyboard focus you opened it with is the
  // reason people stop using the keyboard.
  if (back && back.isConnected && typeof back.focus === 'function') back.focus();
}

/* A Gmail link names the signed-in account it belongs to, and only resolves in
   that browser slot; `gmail_account_index` in config.json says which slot this
   mailbox is. The server already rewrote the stored per-message links with it, so
   a fallback built here has to use the same slot or the two links in one pane
   would disagree about where the mail lives. Shared with the preview box, which
   offers the same way out of the same message. */
function gmailHref(m) {
  const slot = Number.isInteger(state.status.gmail_account_index)
    ? state.status.gmail_account_index : 0;
  if (m.gmail_url) {
    return {
      href: m.gmail_url, label: 'open in gmail',
      title: 'gmail account /u/' + slot + '/',
    };
  }
  return {
    href: 'https://mail.google.com/mail/u/' + slot + '/#search/'
      + encodeURIComponent(m.subject || ''),
    label: 'search gmail for this subject',
    title: 'searches Gmail for: ' + (m.subject || '(no subject)'),
  };
}

/* The inline way out. A note saying "the rest is in Gmail" has to carry the link
   itself: making the reader hunt for the button in the header is a second task. */
function gmailLink(m) {
  const g = gmailHref(m);
  return el('a', { href: g.href, target: '_blank', rel: 'noreferrer',
                   title: g.title }, g.label);
}

/* What the pane has to admit when it shows a mail's HTML rather than its text:
   which part of the message is on screen, whether Gmail was even asked, and what
   was withheld from it. The sanitizer's own sentences about this particular
   message — "12 remote images hidden", "a style this message loads from elsewhere
   is refused here" — go in the same row, because they are the same kind of fact
   and a paragraph per fact is how a reading pane stops being readable. */
function mailNote(data, onImages) {
  /* Every fact its own item, all at one level: the rule draws the separators, and
     a sentence nested inside another one would silently stop getting one. */
  const note = el('div', { class: 'mail-note' },
    el('span', {}, 'this message’s ',
       el('b', { text: data.shown || 'text/html' }), ' part'),
    data.section ? el('span', { text: 'section ' + data.section }) : null,
    data.cached ? el('span', { text: 'cached' }) : null,
    el('span', { text: 'not the part the index searched' }),
    data.charset ? el('span', { text: 'read as ' + data.charset }) : null,
    data.empty ? el('span', { text: 'its html held nothing this app can draw' }) : null);
  for (const n of (data.notes || [])) note.append(el('span', { text: n }));
  if (data.images_allowed || (data.images_blocked || 0) > 0) {
    /* Gmail's own words, because the thing being asked is the same one Gmail
       asks, and "display images" has an meaning every reader already learned
       somewhere else. The answer is a new render, not a permission the mail gets
       to change: the frame's policy is written before the markup is spliced in. */
    note.append(el('button', {
      class: 'link-btn', type: 'button',
      title: data.images_allowed
        ? 'stop loading this message’s images from the servers it names'
        : 'load this message’s images from the servers it names',
      onclick: () => onImages(!data.images_allowed),
    }, data.images_allowed ? 'hide images' : 'display images'));
  }
  return note;
}

/* The body of a mail, and the one question an HTML mail makes you ask of it.
 *
 * Both answers are true and they are different parts of the message: the text is
 * what the index searched and what the highlight you are staring at was drawn
 * from, the HTML is what the sender meant you to see. So the switch names which
 * one is on the screen, in the header row that already said "body", using the same
 * segmented control the search bar uses to say which ranking produced a list —
 * one app, one way of admitting it has two views of a thing.
 *
 * The text is never taken away to make room. The HTML is fetched into the space
 * the text already holds and, if that fails, the mail you were reading is still
 * there with the reason under it: a reading pane that goes blank because a
 * network call went wrong is a reading pane that lost your place.
 */
function bodyBlock(m, toks) {
  const label = m.body_missing ? 'preview — no body indexed' : 'body';
  const wrap = el('div');
  const head = el('h2', { class: 'rb-sec-h', text: label });
  const asMail = el('button', {
    class: 'mode', type: 'button', 'aria-pressed': 'false',
    title: 'show this message’s own html, in a frame that cannot run scripts',
  }, 'as mail');
  const asText = el('button', {
    class: 'mode is-on', type: 'button', 'aria-pressed': 'true',
    title: 'show the text the index searched',
  }, 'as text');
  const seg = el('div', { class: 'modes', title: 'which part of this message to read' },
    asMail, asText);
  const text = el('div', { class: 'body-text' }, highlighted(m.body || '(empty)', toks));
  wrap.append(el('div', { class: 'rb-body-h' }, head, seg), text);

  let view = null, note = null, told = null, seq = 0;

  function clear() {
    seq += 1;
    delete asMail.dataset.busy;
    for (const n of [view, note, told]) if (n) n.remove();
    view = note = told = null;
  }
  function asText2() {
    clear();
    text.hidden = false;
    head.textContent = label;
    asMail.classList.remove('is-on');
    asText.classList.add('is-on');
    asMail.setAttribute('aria-pressed', 'false');
    asText.setAttribute('aria-pressed', 'true');
  }
  function paint(data) {
    clear();
    view = window.MailHtml.frame(data);
    note = mailNote(data, (want) => { state.mailImages = want; show(want); });
    text.hidden = true;
    wrap.append(view, note);
    head.textContent = 'html part';
    asMail.classList.add('is-on');
    asText.classList.remove('is-on');
    asMail.setAttribute('aria-pressed', 'true');
    asText.setAttribute('aria-pressed', 'false');
  }
  function show(images) {
    const mine = seq + 1;
    seq = mine;
    asMail.dataset.busy = '';
    window.MailHtml.load(m.message_id, images).then((data) => {
      if (mine !== seq) return;                    // a slower answer for a faster click
      state.mailImages = !!data.images_allowed;
      paint(data);
    }, (err) => {
      if (mine !== seq) return;
      delete asMail.dataset.busy;
      asText2();
      if (err.reason === 'no_html') asMail.disabled = true;   // nothing to switch to
      told = el('div', { class: 'rb-note warn' }, err.message);
      wrap.append(told);
    });
  }
  asMail.addEventListener('click', () => show(state.mailImages));
  asText.addEventListener('click', asText2);
  return wrap;
}

function renderMessage(m) {
  closeLightbox();
  const body = $('reader-body');
  body.textContent = '';
  const toks = queryTokens(state.q || state.textF || '');
  const d = asDate(m.date);
  const inner = el('div', { class: 'rb-inner' });
  const g = gmailHref(m);

  inner.append(
    el('div', { class: 'rb-top' },
      el('span', { class: 'rb-kicker', text: d ? new Intl.DateTimeFormat('en-GB',
        { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
          hour12: false }).format(d) : 'undated' }),
      el('span', { class: 'rb-kicker', text: '·' }),
      el('span', { class: 'rb-kicker', text: m.folder || '' }),
      el('div', { style: 'margin-left:auto' },
        el('a', { class: 'link-btn', href: g.href, target: '_blank',
                  rel: 'noreferrer', title: g.title }, g.label,
           icon(['M7 13l6-6M8 7h5v5'], { size: 12 })))),
    el('h1', { class: 'rb-subject' }, highlighted(m.subject || '(no subject)', toks)));

  const meta = el('dl', { class: 'rb-meta' });
  meta.append(...kv('from', m.sender || '(no sender)'));
  if (m.to_addrs) meta.append(...kv('to', m.to_addrs));
  if (m.cc) meta.append(...kv('cc', m.cc));
  meta.append(...kv('message', m.message_id, 'mono'));
  if (m.thread_id) meta.append(...kv('thread', m.thread_id, 'mono'));
  inner.append(meta);

  /* The score strip only appears when this message came out of a search: a
     bare "0.71" invites more confidence than the number earns, so it is shown
     as the two ranks it came from, next to the fused value they produced. */
  const hit = state.items.find((x) => x.message_id === m.message_id && x.fused != null) || null;
  if (hit) {
    const num = (v, dp) => (v == null ? '—' : Number(v).toFixed(dp));
    const part = (cls, k, v) => el('span', { class: 'kv' + (cls ? ' ' + cls : '') },
      k + ' ', el('b', { text: v }));
    inner.append(el('div', { class: 'rb-score' },
      part('', 'fused', num(hit.fused, 3)),
      part('dense', 'vector', num(hit.dense, 3) + (hit.dense_rank ? ' #' + hit.dense_rank : '')),
      part('lex', 'keyword', num(hit.bm25, 2) + (hit.bm25_rank ? ' #' + hit.bm25_rank : '')),
      part('', 'prior', num(hit.prior, 2)),
      part('', 'tier', hit.tier || '—'),
      part('', 'matched', hit.label || '—'),
      hit.age_days != null ? part('', 'age', Math.round(hit.age_days) + 'd') : null));
    if (hit.text && hit.text !== m.body) {
      // A label, not an apology: this is the window that scored, and the
      // highlight inside it says why.
      inner.append(el('div', { class: 'rb-note' }, 'the part that matched',
        el('div', { class: 'body-text' }, highlighted(hit.text.replace(/\s+/g, ' ').trim(), toks))));
    }
  }

  const atts = m.attachments || [];
  // Two different truths about the same file: `attachments` lists what the
  // extractor could read, `attachment_exts` lists what arrived. A .txt is
  // metadata-only for the indexer, so the list pane shows "txt" while the
  // reader would otherwise show nothing at all — which reads as a lie.
  const exts = (m.attachment_exts || '').split(',').filter((x) => x && x !== 'none');
  if (atts.length || exts.length) {
    const media = m.media || {};
    const wrap = el('div', { class: 'rb-atts' });
    for (const a of atts) wrap.append(attChip(m, a, media));
    for (const ext of exts.filter((e) => !atts.some((a) => a.toLowerCase().endsWith('.' + e)))) {
      wrap.append(el('span', {
        class: 'att-chip',
        title: 'kept by name and type: this file type is not read for text',
      }, el('span', { class: 'ext', text: ext }), el('span', { text: 'name/type only' })));
    }
    // The chips are the whole story here: a thumbnail says a preview is stored,
    // an extension badge says only the name is, and the icon on each one says
    // whether it opens or downloads. A paragraph explaining that architecture
    // under every mail only repeats what the chips show, and pre-announces a
    // Gmail round-trip that fails, if it fails, in its own sentence.
    const n = Math.max(atts.length, exts.length);
    inner.append(el('div', { class: 'rb-kicker', text: `${n} attachment${n > 1 ? 's' : ''}` }), wrap);
  }

  inner.append(bodyBlock(m, toks));
  if (m.truncated) {
    inner.append(el('p', { class: 'rb-note' },
      'the rest of this mail is in ', gmailLink(m)));
  }
  if (m.quoted) {
    inner.append(el('h2', { class: 'rb-sec-h', text: 'quoted text' }));
    inner.append(el('div', { class: 'body-text quoted' }, highlighted(m.quoted, toks)));
  }
  const th = m.thread || [];
  if (th.length > 1) {
    inner.append(el('h2', { class: 'rb-sec-h', text: `${th.length} in this conversation` }));
    const list = el('div');
    for (const t of th) {
      list.append(el('button', {
        class: 'thread-item' + (t.message_id === m.message_id ? ' is-here' : ''),
        onclick: () => open(t.message_id),
      },
        el('span', { class: 't-when', text: whenLabel(asDate(t.date)) }),
        el('span', { class: 't-from', text: t.sender || '(no sender)' }),
        el('span', { class: 't-subject', text: t.subject || '(no subject)' })));
    }
    inner.append(list);
  }
  body.append(inner);
  $('reader').scrollTop = 0;
}

function closeReader() {
  $('reader').setAttribute('data-closed', '1');
  $('reader-body').hidden = true;
  $('reader-empty').hidden = false;
  state.openId = '';
}

/* ------------------------------------------------------------ selection --- */

function select(i, scroll) {
  const rows = [...document.querySelectorAll('.row')];
  if (!rows.length) return;
  state.sel = Math.max(0, Math.min(rows.length - 1, i));
  rows.forEach((r, j) => r.classList.toggle('is-sel', j === state.sel));
  if (scroll !== false) rows[state.sel].scrollIntoView({ block: 'nearest' });
}

/* --------------------------------------------------------------- wiring --- */

function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

function wire() {
  const q = $('q');
  const onType = debounce(() => {
    state.q = q.value;
    state.textF = '';
    runSearch(false);
  }, 250);
  q.addEventListener('input', () => { state.chips = null; renderChips(); onType(); });
  q.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { state.q = q.value; runSearch(false); }
    if (e.key === 'Escape') { q.blur(); }
  });

  document.querySelectorAll('.mode').forEach((b) => {
    b.addEventListener('click', () => {
      state.mode = b.dataset.mode;
      document.querySelectorAll('.mode').forEach((x) => x.classList.toggle('is-on', x === b));
      runSearch(true);
    });
  });

  // -- search by picture ----------------------------------------------------
  $('img-btn').addEventListener('click', () => $('img-input').click());
  $('img-input').addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    e.target.value = '';                 // so the same file can be picked twice
    setImage(f);
  });
  $('img-clear').addEventListener('click', clearImage);

  // A screenshot sitting in the clipboard is the shortest path from "this thing
  // on my screen" to "which mail is about this thing", so Ctrl+V attaches it.
  // A paste without image files is left completely alone: text stays text.
  document.addEventListener('paste', (e) => {
    const f = e.clipboardData && e.clipboardData.files && e.clipboardData.files[0];
    if (!f || !IMAGE_TYPES.includes(f.type)) return;
    e.preventDefault();
    setImage(f);
  });

  // Drops are handled on the document, not only on the bar: a file dropped
  // anywhere else makes the browser navigate to that file, which throws away the
  // page and the search with it. The bar only lights up to show where to aim.
  const bar = $('searchbar');
  const dragsFiles = (e) => Array.from((e.dataTransfer && e.dataTransfer.types) || [])
    .includes('Files');
  document.addEventListener('dragover', (e) => {
    if (!dragsFiles(e)) return;
    e.preventDefault();
    bar.classList.toggle('is-drop', bar.contains(e.target));
  });
  document.addEventListener('dragleave', (e) => {
    if (!e.relatedTarget) bar.classList.remove('is-drop');
  });
  document.addEventListener('drop', (e) => {
    if (!dragsFiles(e)) return;
    e.preventDefault();
    bar.classList.remove('is-drop');
    setImage(e.dataTransfer.files && e.dataTransfer.files[0]);
  });

  document.querySelectorAll('.view-btn').forEach((b) => {
    b.addEventListener('click', () => {
      setActiveView(b.dataset.view);
      if (b.dataset.view === 'search') {
        state.q ? runSearch(true) : loadBrowse(true);
      } else {
        state.chips = null; renderChips();
        loadBrowse(true);
      }
    });
  });

  $('tune-btn').addEventListener('click', () => {
    const t = $('tuning');
    t.hidden = !t.hidden;
    $('tune-btn').style.color = t.hidden ? '' : 'var(--lamp)';
  });

  const knobChanged = () => {
    state.touched = true;
    state.top = Math.max(1, Math.min(100, parseInt($('t-top').value || '10', 10)));
    state.knobs = {
      min_relevance: parseFloat($('t-min').value),
      agg_k: parseInt($('t-aggk').value || '3', 10),
      recency_half_life_days: parseInt($('t-half').value || '240', 10),
      collapse: $('t-collapse').checked,
      tier_norm: $('t-tiernorm').checked,
      time_weight: $('t-timew').checked,
    };
    state.explain = $('t-explain').checked;
    runSearch(true);
  };
  ['t-top', 't-min', 't-aggk', 't-half'].forEach((id) => {
    $(id).addEventListener('change', knobChanged);
  });
  ['t-collapse', 't-tiernorm', 't-timew', 't-explain'].forEach((id) => {
    $(id).addEventListener('change', knobChanged);
  });
  document.querySelectorAll('.tier').forEach((b) => {
    b.addEventListener('click', () => {
      const t = b.dataset.tier;
      const on = !state.tiers.includes(t);
      state.tiers = on ? state.tiers.concat(t) : state.tiers.filter((x) => x !== t);
      b.classList.toggle('is-on', on);
      runSearch(true);
    });
  });

  $('sender-filter').addEventListener('input', debounce(loadFacets, 200));
  $('more').addEventListener('click', () => loadBrowse(false));

  $('refresh-btn').addEventListener('click', startRefresh);
  $('rp-close').addEventListener('click', () => { $('refresh-panel').hidden = true; });

  const theme = localStorage.getItem('vm-theme') || 'lamp';
  document.documentElement.dataset.theme = theme;
  $('theme-btn').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'lamp' ? 'paper' : 'lamp';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('vm-theme', next);
    $('theme-btn').title = next === 'lamp' ? 'Switch to paper' : 'Switch to lamp';
  });

  document.addEventListener('keydown', (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    const typing = tag === 'input' || tag === 'textarea' || e.target.isContentEditable;
    if ((e.key === '/' && !typing) || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k')) {
      e.preventDefault(); q.focus(); q.select(); return;
    }
    if (e.key === 'Escape') { closeReader(); return; }
    if (typing) return;
    if (e.key === 'ArrowDown' || e.key === 'j') { e.preventDefault(); select(state.sel + 1); }
    else if (e.key === 'ArrowUp' || e.key === 'k') { e.preventDefault(); select(state.sel - 1); }
    else if (e.key === 'Enter' && state.sel >= 0) {
      const rows = [...document.querySelectorAll('.row')];
      if (rows[state.sel]) { open(rows[state.sel].dataset.id); }
    }
  });
}

/* -------------------------------------------------------------- refresh --- */

let refreshTimer = null;

async function startRefresh() {
  const btn = $('refresh-btn');
  try {
    const r = await api('/api/index/refresh', { method: 'POST', body: '{}' });
    $('refresh-panel').hidden = false;
    $('rp-title').textContent = 'indexing — ' + r.cmd;
    btn.classList.add('is-busy');
    btn.disabled = true;
    pollRefresh();
    refreshTimer = setInterval(pollRefresh, 2000);
  } catch (e) {
    toast(String(e.message || e));
  }
}

async function pollRefresh() {
  try {
    const r = await api('/api/index/refresh');
    $('rp-log').textContent = r.log || '';
    $('rp-log').scrollTop = $('rp-log').scrollHeight;
    if (!r.running) {
      clearInterval(refreshTimer);
      refreshTimer = null;
      const btn = $('refresh-btn');
      btn.classList.remove('is-busy');
      btn.disabled = false;
      const code = r.exit_code;
      $('rp-title').textContent = code === 0 ? 'index run finished' : `index run stopped (exit ${code})`;
      // status first: the toast says what the run did, and that is in the
      // status the run just changed, not in the one the poller last fetched.
      await pollStatus();
      // A stopped run needs no toast: the panel directly above the toast spot
      // already carries the exit code, with the log that says why underneath it.
      if (code === 0) toast(refreshDone(r));
      loadFacets();
      if (state.view === 'search' && state.q) runSearch(true); else loadBrowse(true);
    }
  } catch (e) { /* keep polling; the panel is showing the log anyway */ }
}

/* ----------------------------------------------------------------- boot --- */

async function boot() {
  wire();
  closeReader();
  $('t-top').value = state.top;
  $('t-min').value = '0.30';
  $('t-aggk').value = '3';
  $('t-half').value = '240';
  $('t-collapse').checked = true;
  $('t-tiernorm').checked = true;
  document.querySelector('.mode[data-mode="hybrid"]').classList.add('is-on');
  await pollStatus();
  loadFacets();
  loadBrowse(true);
  setInterval(pollStatus, 4000);
  if (!state.status.ready && !state.status.error) {
    $('list-sub').textContent = 'loading the index…';
  }
}

boot();
