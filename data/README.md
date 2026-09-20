# The labelled set

`requests.jsonl` holds 180 user requests, each labelled with the model tier that should answer it and two
properties a router cares about, plus a tag saying how the request was written.

```json
{"id": "sml-11", "message": "What is my current account balance?",
 "tier": "small", "needs_tools": "true", "is_sensitive": "true", "difficulty": "mixed"}
```

## Labelling rules

Labels follow written rules rather than intuition, because a router can only be scored against a policy that
is actually stated somewhere.

- **`tier`** is the cheapest tier that can answer the request *well*.
  - `small` — a lookup, a greeting, a format change, a short rewrite, a one-line answer.
  - `medium` — several steps, ordinary code, a summary that needs judgement, a routine explanation.
  - `powerful` — long multi-step reasoning, specialist knowledge, system design, or consequences in money,
    law, health or safety.
- **`needs_tools`** is true when answering needs information the request does not already contain: a live
  system, private records, or a current fact. It is false when the material arrives with the request, even
  if the request points at it ("explain *this* stack trace"). That second clause is not how the set was first
  labelled: the ambiguity surfaced during the audit and 23 labels were corrected. Even after the rewrite an
  independent blind pass only agrees 0.717 of the time, so treat this question as genuinely contested. See
  [the audit](../docs/EVAL.md#auditing-the-gold-labels-and-the-audit).
- **`is_sensitive`** is true when the *subject matter* carries real-world consequences in money, law, health
  or safety.

## The one deliberate trap

`is_sensitive` is about the **topic**; `tier` is about the **task**. They are labelled independently on
purpose, because confusing the two is the routing failure that costs real money.

> *"What is my current account balance?"* — sensitive (money), but it is a lookup: `small`.
> *"Should I accept a settlement of 40 thousand or go to tribunal?"* — sensitive **and** `powerful`.

8 of the 47 sensitive requests are `small` and 3 are `medium`. A router that has quietly learnt "money or law
mentioned ⟹ send it to the expensive model" will get those 11 wrong, and will do so on exactly the requests
that are cheapest to answer. The evaluation reports overspend and underspend separately so this shows up.

## Composition

| tier | n | | difficulty | n | what it is |
|---|---:|---|---|---:|---|
| `small` | 63 | | `clear` | 118 | One dominant signal, written plainly |
| `medium` | 56 | | `terse` | 22 | A handful of words, sometimes one |
| `powerful` | 61 | | `mixed` | 18 | Surface signals argue for the wrong tier |
| | | | `negated` | 13 | Says it is simple when it is not, or vice versa |
| | | | `noisy` | 9 | Typos, shouting, rambling |

`needs_tools` is true for 24 requests after the correction described above, and still appears at every tier,
so it cannot be inferred from the tier.

## Honest limitations

- **The requests are written, not collected.** Real prompt logs are not public, and a synthetic set written by
  one person can encode that person's idea of the task. `laya-router eval adjudicate` has GPT-5 label the set
  **blind** and reports agreement: 0.817 on `tier`, 0.944 on `is_sensitive`, 0.717 on `needs_tools`
  (`results/adjudication.jsonl`). So the ceiling on the routing decision is about 0.82, not 1.0.
- **One annotator.** There is no second human pass, so there is no inter-annotator agreement to quote.
- **The tier boundaries are a judgement call.** "Cheapest tier that can do it well" is a policy, not a fact;
  a different organisation would draw the `medium`/`powerful` line elsewhere. That is precisely why the
  evaluation measures a router against a *stated* policy rather than against a notion of correctness.
- **180 requests.** Wide enough to break down by tier and difficulty, narrow enough that differences smaller
  than about 7 points are inside the noise.
- **English only.** The Laya checkpoint used here is the English one; the SDK ships a multilingual checkpoint
  that is not evaluated.
