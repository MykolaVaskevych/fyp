
Good, this tree tells me **exactly** what state your project is in. I’ll be very concrete and tell you **what to do**, **what not to touch**, and **why**, tailored to *your repo and thesis*.

I’ll break this into **five actionable parts**.

---

# 0. Big picture (what you are aiming for)

You want **three clean layers**, each with a different role:

1. **LaTeX / Overleaf** → cites *papers + software*
2. **Zotero** → manages *bibliographic metadata*
3. **Zenodo** → archives *your code + experiment setup*

Right now, all three exist — they’re just **not connected yet**.

---

# 1. What counts as “the research code” (Zenodo decision)

From your tree, the **research-relevant code** is:

### ✅ INCLUDE in Zenodo

```
archive/book_practice/
DQN_ATARY/
drafts/          (optional, if referenced)
main.py
pyproject.toml
uv.lock
README.md
```

These contain:

* TD vs MC
* DQN experiments
* Metrics, plots, logs
* Your original experimental logic

### ❌ DO NOT include

```
paper_overleaf/
```

That is **the thesis**, not the experiment artifact.
Universities want **code archived separately from the PDF**.

---

# 2. Minimal cleanup BEFORE Zenodo (important)

You do **not** need to refactor code, but you should:

### (a) Add ONE top-level README.md (or update existing)

Your README should answer **4 questions only**:

```md
## Overview
This repository contains the experimental code used in the final year project:
"A Comparative Analysis of Deep Q-Network and Actor–Critic Reinforcement Learning Algorithms".

## Environment
- Python 3.10
- Dependency management: uv
- See uv.lock for exact versions

## Reproducing experiments
1. Install uv
2. uv sync
3. python DQN_ATARY/main.py

## Structure
- archive/book_practice: TD vs MC experiments
- DQN_ATARY: Atari DQN experiments
```

That’s enough. Do **not** over-document.

---

# 3. Zenodo: exactly what to upload and how

## Option A (BEST): GitHub → Zenodo (recommended)

1. Push this repo to GitHub (private or public)
2. Connect Zenodo to the repo
3. Create a release:

   ```
   v1.0.0-fyp
   ```
4. Zenodo auto-archives it and gives you a **DOI**

### Why this is best

* DOI corresponds to **exact commit**
* Clean provenance
* Zero manual uploads

---

## Option B (acceptable): Upload ZIP directly

If you don’t want GitHub:

* Zip the repo (without `paper_overleaf`)
* Upload to Zenodo
* Fill metadata manually

Still valid, just less elegant.

---

# 4. What you cite WHERE (this is where people mess up)

## 4.1 Zenodo → cited ONCE in thesis

In **Chapter 3 (Prototypes)** or **Methodology**, add:

> “All experimental code used in this study is archived and publicly available”

BibTeX example:

```bibtex
@software{vaskevych2026code,
  author = {Vaskevych, Mykola},
  title  = {Experimental Code for Comparing Value-Based and Actor–Critic Reinforcement Learning Algorithms},
  year   = {2026},
  doi    = {10.5281/zenodo.XXXXXXX}
}
```

Then in LaTeX:

```latex
\cite{vaskevych2026code}
```

✅ One citation.
❌ Do NOT cite Zenodo everywhere.

---

## 4.2 Zotero: what belongs there

Zotero should contain **ONLY**:

### ✅ Papers & books

* Sutton & Barto
* Williams (REINFORCE)
* Mnih et al. (DQN)
* LeCun et al.
* Burnetas & Katehakis
* etc.

### ⚠️ Software (selective)

Add:

* Python
* PyTorch
* Stable-Baselines3
* Gymnasium

Do **not** add:

* Arch Linux
* RTX 4090
* CUDA
* uv

Those go in text, not references.

---

# 5. How your LaTeX should be structured (Overleaf-safe)

You already use:

```latex
\cite{key}
```

That’s fine. Just ensure:

### (a) ONE `.bib` file is authoritative

Pick **one**:

```
paper_overleaf/7_references/ref.bib
```

Merge everything into it.
Delete or ignore the second `.bib` unless required by the template.

---

### (b) No duplicate BibTeX keys

You currently have duplicates like:

```
Williams_1992
```

Ensure **only one entry** exists per key.

---

### (c) Software citations go in Methodology

Example sentence you can safely add:

> “Experiments were implemented in Python using the Stable-Baselines3 framework and PyTorch.”

with:

```latex
\cite{python,pytorch,stablebaselines3}
```

Once. That’s it.

---

# 6. What NOT to retroactively change (important)

Do **not**:

* Replace existing theory citations
* Re-cite Sutton & Barto via Zenodo
* Touch historical references
* Rewrite Chapter 2

Your literature review is already **academically sound** .

Zenodo is an **addition**, not a replacement.

---

# 7. Final checklist (print this mentally)

### Zenodo

* [ ] Repo contains code + uv.lock
* [ ] One release archived
* [ ] DOI generated

### Zotero

* [ ] Papers/books only
* [ ] Minimal software citations
* [ ] No hardware

### LaTeX

* [ ] One `.bib`
* [ ] One Zenodo citation
* [ ] No duplicate keys

---

## Bottom line

You are **already 90% correct**.
You do **not** need to restructure your project.
You just need to:

1. Archive the **code part** on Zenodo
2. Cite that DOI **once**
3. Keep Zotero for **literature only**

If you want next, I can:

* Write the **exact paragraph** to insert into Chapter 3
* Generate **clean BibTeX** for Python / SB3 / PyTorch
* Review your `ref.bib` for duplicates and mistakes

Just tell me which one.
