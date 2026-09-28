"""Citation pills in static/app.js select the source card the tag actually cites.

Runs the real app.js in Node with minimal DOM stubs. Skipped when Node is
not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "static" / "app.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

# Loads app.js in a sandbox, renders the answer's pills, clicks each one and
# reports which source card became active.
_HARNESS = r"""
const vm = require("vm");
const fs = require("fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const el = () => ({ style: {}, classList: { add() {}, remove() {}, contains: () => true }, textContent: "", innerHTML: "" });
const cards = input.retrieved.map((_, i) => ({
  idx: i, active: i === 0,
  classList: { toggle(cls, on) { if (cls === "active") this.owner.active = on; } },
  scrollIntoView() {},
}));
cards.forEach(c => { c.classList.owner = c; });
const ctx = {
  document: {
    getElementById: () => el(),
    querySelectorAll: () => cards,
    querySelector: sel => cards[Number(sel.match(/data-idx="(\d+)"/)[1])],
  },
  localStorage: { getItem: () => null, setItem() {} },
  fetch: () => Promise.reject(new Error("offline")),
  setInterval: () => 0,
  console,
};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(input.appJs, "utf8"), ctx);
const html = ctx.formatAnswerWithCitations(input.answer, input.retrieved);
const pills = [...html.matchAll(/<span class="cite"(?: onclick="focusSource\((\d+)\)")?[^>]*>([^<]*)<\/span>/g)];
const out = pills.map(m => {
  const clicked = m[1] === undefined ? null : Number(m[1]);
  if (clicked !== null) ctx.focusSource(clicked);
  return { label: m[2], active: clicked === null ? null : cards.findIndex(c => c.active) };
});
process.stdout.write(JSON.stringify(out));
"""


_MATCH_HARNESS = r"""
const vm = require("vm");
const fs = require("fs");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const el = () => ({ style: {}, classList: { add() {}, remove() {}, contains: () => true }, textContent: "", innerHTML: "" });
const ctx = {
  document: { getElementById: () => el(), querySelectorAll: () => [], querySelector: () => null },
  localStorage: { getItem: () => null, setItem() {} },
  fetch: () => Promise.reject(new Error("offline")),
  setInterval: () => 0,
  console,
};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(input.appJs, "utf8"), ctx);
const out = input.cases.map(([allowed, cited]) =>
  ctx.matchChunkIndex([{ paper_title: allowed, page: 3, section: "methods" }], cited, "3", "other"));
process.stdout.write(JSON.stringify(out));
"""


def _chunk(title: str, page: int, section: str) -> dict:
    return {"paper_title": title, "page": page, "section": section}


RETRIEVED = [
    _chunk("Neural Collaborative Filtering", 1, "abstract"),
    _chunk("Neural Collaborative Filtering", 5, "experiments"),
    _chunk("BPR: Bayesian Personalized Ranking from Implicit Feedback", 2, "related_work"),
    _chunk("BPR: Bayesian Personalized Ranking from Implicit Feedback", 4, "methods"),
    _chunk("Eﬀects using Random Forests", 3, "related_work"),
]


def _click_pills(answer: str) -> list[dict]:
    proc = subprocess.run(
        ["node", "-e", _HARNESS],
        input=json.dumps({"appJs": str(APP_JS), "answer": answer, "retrieved": RETRIEVED}),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(proc.stdout)


def test_first_pill_can_point_to_a_later_card() -> None:
    answer = (
        "BPR optimizes a pairwise criterion "
        "[Paper: BPR: Bayesian Personalized Ranking from Implicit Feedback | p.4 | §methods]. "
        "NCF uses log loss [Paper: Neural Collaborative Filtering | p.5 | §experiments]."
    )
    # First pill in the text cites card 4, second cites card 2.
    assert _click_pills(answer) == [{"label": "4", "active": 3}, {"label": "2", "active": 1}]


def test_ligature_title_resolves_to_its_card() -> None:
    answer = "Honest trees [Paper: Effects using Random Forests | p.3 | §related_work]."
    assert _click_pills(answer) == [{"label": "5", "active": 4}]


def test_shortened_title_resolves_on_same_page_only() -> None:
    answer = "Pairwise [Paper: Bayesian Personalized Ranking | p.4 | §other]."
    assert _click_pills(answer) == [{"label": "4", "active": 3}]


def test_unmatched_tag_gets_no_click_target() -> None:
    answer = "Short spoof [Paper: BPR | p.2 | §related_work] and wrong page [Paper: Neural Collaborative Filtering | p.9 | §x]."
    assert _click_pills(answer) == [{"label": "?", "active": None}, {"label": "?", "active": None}]


# (allowed title, cited title). Non-ASCII letters next to the cited span decide
# the whole-word rule; the UI must agree with rag/guards/citation_checker.py.
PARITY_CASES = [
    ("Über Modelle für Straßen", "Modelle für"),  # whole words
    ("Über Modelle für Straßen", "ber Modelle"),  # preceded by the letter Ü
    ("Straßenbahn Netze Modelle", "Straßen"),  # followed by the letter b
    ("数据 Deep Learning 模型", "Deep Learning"),  # CJK words separated by spaces
    ("模型Deep Learning", "Deep Learning"),  # preceded by a CJK letter
    ("Café Society Models", "é Society"),  # preceded by the letter f
    ("Eﬀects using Random Forests", "Effects using"),  # ligature after NFKC
    ("Deep Learning Models", "Deep"),  # below the 6-character floor
]


def test_short_title_word_boundaries_match_python_checker() -> None:
    from rag.guards.citation_checker import validate_citations_against_context

    python = [
        validate_citations_against_context(
            [f"[Paper: {cited} | p.3 | §other]"], [f"[Paper: {allowed} | p.3 | §methods]"]
        ).all_valid
        for allowed, cited in PARITY_CASES
    ]
    proc = subprocess.run(
        ["node", "-e", _MATCH_HARNESS],
        input=json.dumps({"appJs": str(APP_JS), "cases": PARITY_CASES}),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    ui = [idx >= 0 for idx in json.loads(proc.stdout)]
    assert ui == python
    assert python == [True, False, False, True, False, False, True, False]
