# Privacy Notice

ForensicAI processes video footage (including CCTV) to detect people and
objects, build event timelines, and re-identify individuals across frames.
Because it analyzes images of people, it handles **sensitive personal and
biometric information**. This document describes how the software handles that
data so operators can deploy it responsibly.

> **Not legal advice.** This notice describes the software's technical
> behavior. You (the deployer/operator) are the data controller and are
> responsible for your own legal compliance, notices, consent, signage, and
> data-handling policies. Adapt this document before relying on it.

---

## 1. Local-only processing

**Video analysis runs entirely on the machine you run it on.** The computer
vision pipeline — object detection (YOLOv8), tracking, and person
re-identification (ResNet feature embeddings) — executes locally using
PyTorch/OpenCV. Uploaded videos and analysis results are **not** sent to any
third party by the analysis pipeline.

### Important exception: the chat assistant

The optional "Chat with the Case AI" feature can use a **Hugging Face
Inference API** call. When `HF_API_TOKEN` is configured, the chat feature
sends a text prompt — which includes **derived case context** (e.g. number of
persons detected, object/timeline summaries, the case name) — to Hugging Face's
servers to generate a response. This text is governed by Hugging Face's privacy
practices, not by the local-only guarantee above.

- The chat feature **does not** transmit the source video or image crops.
- If `HF_API_TOKEN` is **unset**, the chat runs in local "demo mode" and makes
  no external calls. For a strictly local-only deployment, leave the token
  unset.

No analytics, telemetry, or crash reporting is sent anywhere by this software.

---

## 2. What data is collected and stored

All data is stored on the local filesystem under the `data/` directory:

| Data | Location | Description |
|------|----------|-------------|
| Uploaded videos | `data/uploads/` | The original footage you provide. |
| Analysis results | `data/analysis/<case_id>/` | JSON results, event timelines, detection metadata. |
| Person crops / thumbnails | `data/analysis/<case_id>/` | Cropped images of detected individuals and re-ID galleries. |
| Biometric embeddings | within analysis JSON | Numeric appearance feature vectors used to match a person across frames. |
| Job records | `data/jobs.db` (SQLite) | Processing status/metadata per case. |

### Biometric data

Person re-identification produces **biometric identifiers** — appearance-based
feature embeddings and image crops that can be used to recognize a specific
individual. Under some laws (see §4) these are regulated biometric information.
Treat the `data/analysis/` directory and embeddings as sensitive.

---

## 3. Data retention and deletion

- **Retention:** This software does **not** automatically delete uploaded
  videos, analysis output, biometric embeddings, or thumbnails. They remain on
  disk until you remove them. There is currently no built-in retention timer.
- **Deletion:** To delete a case and its biometric data, remove the relevant
  directory:
  - Source video: delete the file in `data/uploads/`.
  - All derived/biometric data for a case: delete `data/analysis/<case_id>/`.
  - Job record: remove the row from `data/jobs.db` (or delete the file to clear
    all job history).
- **Operator responsibility:** You should define and enforce a retention
  schedule appropriate to your jurisdiction and purpose (e.g. delete biometric
  embeddings when no longer needed for the stated purpose). Keeping biometric
  data longer than necessary increases legal and breach risk.

---

## 4. Regulatory notes

These are high-level pointers, **not** an exhaustive compliance checklist.

### CCPA/CPRA (California)

- Footage and derived data that identify or could be linked to a person are
  "personal information"; biometric embeddings are "sensitive personal
  information" under the CPRA.
- If you are a covered business, obligations may include: a privacy notice at
  or before collection, honoring consumer rights (access, deletion,
  correction, opt-out of certain uses), purpose limitation, and reasonable
  security.
- Recording people in some contexts may also implicate California recording/
  surveillance laws independent of ForensicAI.

### BIPA (Illinois Biometric Information Privacy Act)

- Face/appearance-based identifiers used to identify an individual can qualify
  as "biometric identifiers"/"biometric information."
- BIPA generally requires, **before collection**: a written, published
  retention-and-destruction policy; informed **written consent** from each
  individual; and prohibits selling/profiting from biometric data. It also
  requires destruction when the purpose is satisfied (or within statutory
  limits) and reasonable safeguards.
- BIPA includes a **private right of action** with statutory damages, so
  non-compliance carries significant exposure. Obtain consent and publish a
  retention/destruction policy before processing footage of identifiable
  individuals in scope.

### Other

Other jurisdictions have analogous rules (e.g. GDPR/UK GDPR treat biometric
data used for unique identification as a special category requiring a lawful
basis). Evaluate the laws applicable to where the footage was captured and
where individuals reside.

---

## 5. Security considerations

- The API requires an `X-API-Key` header on `/api/*` requests and restricts
  CORS to configured origins, but it does **not** by itself encrypt stored
  data. Protect the `data/` directory with filesystem permissions and
  full-disk/volume encryption.
- Do not expose the service directly to the public internet without a
  hardened reverse proxy, TLS, and authentication appropriate to the
  sensitivity of the footage.

---

## 6. Summary

- Video analysis is local-only; the optional Hugging Face chat is the sole
  external data path and is off when `HF_API_TOKEN` is unset.
- Biometric embeddings and person crops are sensitive and are retained until
  you delete them.
- If you process footage of identifiable individuals, review CCPA/CPRA, BIPA,
  and any local laws — and obtain consent and publish retention policies where
  required.

_Last updated: 2026-06-21. This is a template; review and adapt for your
deployment._
