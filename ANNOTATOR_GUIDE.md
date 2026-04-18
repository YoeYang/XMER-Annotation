# Annotator Guide: Multimodal Emotion Annotation
### Appraisal-Based Analysis of Complex Emotions in Video

---

## Welcome

Thank you for being part of this annotation project. Before you start, please take 10–15 minutes to read this guide carefully. It will make the whole process much smoother — and your annotations much more valuable.

This project is different from typical emotion labeling tasks. We are **not** asking you to simply put a label like "happy" or "sad" on a video clip. Instead, we want to understand *why* a person feels what they feel, and *why* their face, voice, and words sometimes send conflicting signals.

---

## What We're Actually Annotating

<!-- Every video clip shows a person expressing something — but the expression across their **face**, **voice**, and **words** doesn't always tell the same story.

For example:
- Someone says *"I'm fine"* in a flat, low voice while their brow is furrowed and eyes are averted.
- Someone says *"He's going to be the best kid in school!"* with a furrowed brow and tense jaw, but a forceful, unwavering voice.

These **mismatches between modalities** are the heart of what we're studying. They reveal something genuinely interesting about how complex emotions work — and current AI systems struggle with them badly.

Your job is to help us understand these mismatches, not just detect them. -->

Every video clip shows a person in the middle of an emotional moment — reacting to something that just happened, processing something they just heard, or expressing something they've been holding in.

Your core task is **empathy-driven appraisal reconstruction**: put yourself in the speaker's shoes, and try to simulate the evaluations they are making about their situation. Not what *you* would feel in that situation — what *they* are likely feeling, given their goals, their relationships, and their circumstances as shown in the clip and context card.

---

## The Theoretical Framework (Plain Version)

You don't need to be a psychologist to annotate well. But understanding the basic logic behind our annotation protocol will help you make better judgments.

### Emotions are not states — they are processes

Think about the last time you felt genuinely angry. You didn't just "snap into anger." Something happened, you noticed it was relevant to you, you figured out it was bad for your goals, you realized someone was responsible, and you felt like you could push back. That whole chain of evaluation — happening mostly automatically, in under a second — is what produces the emotion.

This is the core insight of **appraisal theory**: emotions arise from how we evaluate events, not from the events themselves. Two people can experience the same event and feel completely different things, because their evaluations differ.

### The six appraisal dimensions (Smith & Ellsworth, 1985)

These are the key "evaluation questions" the mind asks when processing an emotional event:

| Dimension | The question being asked | Why it matters |
|---|---|---|
| **Pleasantness** | Is this good or bad for me? | The most fundamental split — positive vs. negative emotion |
| **Certainty** | Do I know what's happening / what will happen? | Distinguishes fear and hope (uncertain) from anger and sadness (certain) |
| **Attentional Activity** | Do I want to engage or avoid this? | Disgust and boredom → avoid; interest and challenge → engage |
| **Anticipated Effort** | Does this require effort from me? | Challenge and frustration → high effort; joy and surprise → low effort |
| **Agency** | Who caused this — me, someone else, or circumstances? | Guilt → self; anger → other; sadness → circumstance |
| **Situational Control** | Can anyone change what's happening? | Anger → yes (other has control); sadness → no (no one can) |

**Practical insight from the research:** Pleasantness is the most powerful dimension for telling emotions apart at a broad level. But for distinguishing *similar* emotions (like anger vs. contempt, or shame vs. guilt), the smaller dimensions matter most.

Some emotion pairs that are easy to confuse and what separates them:
- **Anger vs. Sadness**: both unpleasant, but anger = high control by other person; sadness = low control by anyone
- **Fear vs. Anger**: both negative, high urgency, but fear = low coping power; anger = high coping power
- **Shame vs. Guilt**: almost identical appraisal profiles — both self-caused, unpleasant, effortful — but shame involves a slightly lower certainty and stronger avoidance of attention
- **Interest vs. Hope**: both pleasant, but interest has strong attentional pull and high certainty; hope has moderate uncertainty about whether the desired outcome will occur
- **Boredom**: uniquely characterized by very low anticipated effort and a strong desire to *disengage* — almost the opposite of most other emotions

### Why modalities don't always agree (Scherer's CPM, 2009)

Here's the key insight for understanding why the clips you'll annotate look the way they do.

The face, voice, and language don't all update at the same time. When something happens, the brain processes it in a sequence of evaluations (*Stimulus Evaluation Checks*, or SECs), each completing at a different time:

```
Event occurs
    ↓
Novelty check          (~90ms)   → face shows surprise / orienting response
Pleasantness check     (~100ms)  → face shows approach or rejection
Goal relevance check   (~130ms)  → sustained facial expression begins
Goal conduciveness     (~500ms)  → voice energy changes, body responds
Coping potential       (~600ms)  → voice power/weakness becomes clear
Norm compatibility     (~800ms+) → language filters what gets said
```

This means:
- **The face** often reflects the earliest, most automatic evaluations (is this pleasant? is this novel?)
- **The voice** reflects the slightly later evaluations (can I handle this? is this blocking my goals?)
- **Language** is the most filtered — it reflects the final, socially-regulated expression of the emotion

**This is why mismatches happen — and they are not random.** A person who says "I'm fine" while their face shows distress is likely expressing a real negative appraisal (face) that is being overridden by social norms (language). A person with a furrowed brow but a powerful, confident voice is likely showing that they *recognize an obstacle* (brow) while simultaneously *feeling capable of overcoming it* (voice) — which together produces something like **Determination**.

We are *not* asking you to annotate the millisecond-level neural process. We are asking you to look at the **output traces** left in face, voice, and language, and reason about which appraisal evaluations they reflect.

<!-- ---

## Types of Modality Conflict You'll Encounter

As you annotate, you'll notice the conflicts tend to fall into recognizable patterns:

**1. Emotional Regulation / Suppression**
The person's true feeling leaks through one modality (usually face or voice) but is masked in another (usually language). Very common in professional or social settings.
*Example: Someone who just received bad news says "That's totally fine" with a tense jaw and tight lips.*

**2. Genuine Complex Emotion (SEC Co-display)**
Both signals are authentic — the person is genuinely experiencing two different appraisal evaluations simultaneously. The "contradiction" reflects the complexity of the emotion, not a performance.
*Example: The "He'll be the best kid in school!" clip — furrowed brow (obstacle detected) + powerful voice (high coping power) = Determination.*

**3. Emotion in Transition**
The person is moving from one emotional state to another. Different modalities update at different speeds, so you catch them mid-transition.
*Example: Face still shows confusion while voice has shifted to frustration.*

**4. Mixed Emotion**
Two genuinely different emotional states coexist — typically triggered by a single event that has conflicting implications.
*Example: Pride mixed with guilt after succeeding at something through questionable means.* -->

---

## A Note on Individual Differences

You'll notice that for some clips, what seems like a "mismatch" might actually reflect something about the specific person — their personality, cultural background, or emotional regulation habits. For instance, a perfectionist tends to overestimate the relevance of every obstacle (which can make their frustration look disproportionate), while someone with low self-esteem tends to underestimate their own coping ability (which makes fear look stronger than the situation warrants).

You don't need to diagnose anyone. But keeping this in mind will help you not over-pathologize expressions that might just reflect individual style.

---

## Recommended Reading

We strongly recommend reading the following two papers before you begin. You don't need to read every word — the sections highlighted below are the most relevant.

**Paper 1 — The foundation of our appraisal dimensions:**
> Smith, C. A., & Ellsworth, P. C. (1985). Patterns of cognitive appraisal in emotion. *Journal of Personality and Social Psychology, 48*(4), 813–838.

Focus on: the introduction (why these six dimensions), Table 3–4 (the appraisal profiles of each emotion), and the discussion of emotion pairs.

**Paper 2 — Why modalities diverge and how emotion unfolds over time:**
> Scherer, K. R. (2009). The dynamic architecture of emotion: Evidence for the component process model. *Cognition and Emotion, 23*(7), 1307–1351.

Focus on: **Figure 2** (the time-sequence of appraisal checks driving each response modality — this is the most important figure in the paper), **Table 1** (which appraisal check predicts which facial/vocal/physiological response), and **Figure 5** (the timeline of when each check completes). The individual differences section is optional but interesting.

Phoebe Ellsworth's influence runs through both papers — she co-authored the S-E framework with Craig Smith, and her work on appraisal theory across the 1980s–2000s was central to the field. These two papers together represent the theoretical backbone of everything you'll be doing in this annotation task.

---

## Before You Start Annotating Each Clip

1. **Watch the clip once** without trying to analyze anything. Just experience it.
2. **Read the context card** for that clip. Understanding the person's situation and goals is essential for judging appraisal dimensions like goal conduciveness and coping potential.
3. **Watch the clip a second time**, paying attention to face, voice, and language separately.
4. **Then fill in the annotation form.**

If you feel genuinely uncertain about a dimension, always use the **"Unsure"** option rather than guessing. Uncertainty itself is informative data — if multiple annotators mark the same dimension as uncertain, it tells us the signal in that modality is genuinely weak for that clip.

---

## Annotation Quick Reference

**When you see...** → **Think about...**

| Observable signal | Likely appraisal reflection |
|---|---|
| Raised inner brow + drooping lids | Low coping power (sadness, helplessness) |
| Furrowed brow + narrowed eyes + tense jaw | Obstacle detected, but high power (determination, anger) |
| Wide eyes + raised brows | Novelty check firing (surprise, fear) |
| Lip corner pull (smile) | Pleasantness positive (joy, relief) |
| Nose wrinkle + upper lip raise | Rejection response (disgust, contempt) |
| High energy, loud voice | High ergotropic activation — goal blocked but power high (anger, determination) |
| Low energy, lax voice | Low coping potential (sadness, helplessness, resignation) |
| Thin, tense voice (high pitch + restricted range) | Fear profile — low control, high threat |
| Saying positive things in flat/low voice | Language filtered by norms; voice leaking real appraisal |

---

*Thank you for your careful attention. The quality of this dataset depends on your thoughtful judgment.*

*Questions? Please reach out before submitting rather than after.*
