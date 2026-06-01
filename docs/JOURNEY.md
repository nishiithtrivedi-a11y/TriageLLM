# Build notes — TriageLLM

These are the notes from building TriageLLM: the one problem that actually
shaped the design, the numbers behind the fix, and a few smaller things I
learned. I'm not a software developer by training — I work in SAP supply chain
consulting and built this as a side project with a lot of help from AI coding
assistants. So read this as a hobbyist's field notes, not an expert reference.
If something here is wrong or could be done better, please open an issue.

---

## What I started with

A laptop with a 12 GB GPU (RTX 5070), 32 GB RAM, and Ollama already set up with
a few coder models — from `qwen2.5-coder:1.5b` (1 GB) up to a couple of 24 GB
custom builds, plus a tiny `qwen2.5:0.5b` for the classifier and critic.

The goal was simple: stop paying cloud-API rates for "rename this variable" and
"write a quick regex" when my laptop can handle them, and only reach for the
cloud on problems the local models genuinely can't do. The plan was a proxy that
classifies each prompt, routes it to the smallest model that can handle it,
checks the answer, and escalates only when needed.

---

## The one problem worth writing down: the eviction cascade

This is the only discovery interesting enough to document in detail, because it
took a while to understand and the fix isn't obvious.

I sent a normal medium-difficulty prompt through the proxy. It correctly picked
tier M and called a 9 GB model, which loaded into my 12 GB GPU and used almost
all of it. The critic — only 400 MB — also needed the GPU to grade the answer.
But by the time the worker finished writing, **the critic had been evicted from
VRAM** to make room for the worker. When the critic tried to grade, it had to
reload from disk, which took longer than my timeout, so the critic call returned
nothing.

My code's rule was: no score means "treat as bad" and escalate. So it escalated
to a bigger model (which doesn't even fit — it spills into system RAM), the
critic timed out *again*, and it escalated once more. I watched the proxy log
cascade through tiers I didn't need for five minutes while the request hung.

The bug wasn't in any one line. It was a cross-layer interaction: Ollama's
eviction policy + where I put the critic + how my code interpreted a timeout. My
code mistook **"the grader was slow"** for **"the answer was bad."**

## How I fixed it

The fix that mattered: **pin the critic to the CPU** (`num_gpu: 0`). It's a 0.5B
model, small enough that CPU runs it in about 0.6 seconds — and because it never
touches the GPU, nothing can evict it. The GPU is left entirely to the worker
model.

I added a few smaller safeguards around that:

- **Keep the critic loaded** for the whole proxy session (`keep_alive: -1`) so it
  never pays a cold reload.
- **Warm it up at startup** so the first real request isn't slow.
- **Tier-aware soft-pass:** if the critic still somehow returns no score, ship the
  answer for the small tiers (S/M) instead of escalating; only escalate on the
  bigger tiers where a second opinion is worth the cost.

The real change, though, was conceptual: the code now treats "failed to grade
because something broke" and "failed to grade because the answer was bad" as
**different events**. That distinction is what stops the cascade.

---

## Numbers from this machine

Measured on my laptop (Ryzen 7 260 + RTX 5070 12 GB + 32 GB DDR5) with the
`benchmark.py` script in this repo:

| Operation | Median | p95 |
|---|---|---|
| Critic call (CPU-pinned) | 0.58 s | 0.63 s |
| Critic call (GPU-pinned) | 0.58 s | 0.58 s |
| Tier S query (1.5B, warm) | ~3-4 s | — |
| Tier M query (16B, warm) | ~5-10 s | — |

The point: at this model size the CPU critic is the same latency as the GPU
critic, but it's immune to the eviction problem above. That's the whole reason
the design works on a 12 GB card. Your numbers will differ — run `benchmark.py`
on your own hardware.

---

## A few smaller things I learned

- **Small classifiers should advise, not decide.** I run a rule-based classifier
  first, then a 0.5B LLM classifier — but the LLM is only allowed to *escalate*
  the tier the rules chose, never lower it. Left to itself the 0.5B model would
  confidently route a multi-file refactor to the smallest model.
- **`from __future__ import annotations` + `@dataclass` + LiteLLM crashes at
  startup.** LiteLLM loads the hook module via `spec_from_file_location`, and with
  PEP 563 string annotations the dataclass machinery can't resolve the module and
  throws `AttributeError: 'NoneType' object has no attribute '__dict__'`. The fix
  is just to not use the future import — Python 3.10+ handles `dict[X, Y]` natively.
- **Streaming critique can only append, not replace.** Chunks already streamed to
  the client can't be retracted, so a bad-answer verdict mid-stream becomes "append
  a handoff note at the end," not "swap the answer."
- **A bit of Windows setup friction** (the console code page, a stale venv after
  moving machines, Ollama defaulting to the wrong model folder) — all one-line
  fixes, none of them interesting enough to detail here.

---

## Closing

I built this to save money on routine API calls, and the part that turned out to
matter wasn't the routing — it was the failure handling, especially how a slow
critic gets misread as a bad answer. If you run TriageLLM on different hardware
and hit something I didn't, an issue would be welcome. For who built it and why,
see the "About the author" section in the [README](../README.md).
