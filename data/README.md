# The labelled set

`requests.jsonl` holds 150 inbound support messages, each with three gold labels and a tag saying how the
message was written.

```json
{"id": "bil-01", "message": "I was charged twice for invoice 4411 this morning. Please refund the duplicate charge.",
 "queue": "billing", "urgent": "false", "needs_human": "true", "difficulty": "clear"}
```

## Labelling rules

The two yes/no labels follow written rules rather than intuition, because a model can only be scored against a
policy that is actually stated somewhere.

- **`urgent` is true** when the message describes something already broken, work that is blocked right now, or a
  deadline the customer names. Annoyance is not urgency; a question about future plans is not urgency.
- **`needs_human` is true** when an automated reply would be wrong: money moving (refunds, chargebacks,
  cancellations with money attached), legal or regulatory matters, account security and compromise, and
  customers in evident distress. Everything a good help-centre article could answer is false.

`queue` is the team that should own the message. Where a message touches two teams, the label is the team that
must act, not the team the words sound like.

## Difficulty tags

The mix is deliberate, so results can be broken down instead of averaged into one uninformative number.

| Tag | n | What it is |
|---|---:|---|
| `clear` | 86 | One dominant signal, written plainly |
| `terse` | 20 | A handful of words, sometimes one |
| `mixed` | 20 | Two queues in play, or a policy call that the surface words argue against |
| `negated` | 13 | Contains the keyword for the wrong answer, negated (`I am NOT asking for a refund`) |
| `noisy` | 11 | Typos, shouting, rambling |

## Honest limitations

- **The messages are written, not collected.** Real support inboxes are not public, and a synthetic set written
  by one person can encode that person's idea of the task. `laya-router eval adjudicate` re-checks every label
  with a stronger, independent model and reports the disagreements; `results/adjudication.jsonl` is the record.
- **One annotator.** There is no second human pass, so there is no inter-annotator agreement to quote.
- **150 messages.** Wide enough to break down by difficulty, narrow enough that a few points of accuracy are
  inside the noise. Differences smaller than about 8 points should not be read as real.
- **English only.** The Laya checkpoint used here is the English one; the SDK ships a multilingual checkpoint
  that is not evaluated.
