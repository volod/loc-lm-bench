# LLM security learning path

An extended syllabus for evaluating and securing LLM applications, especially local RAG and
tool-using systems. This guide expands [Stage 9 of the main learning path](learning-path.md).

The goal is not to collect clever attack prompts. The goal is to build a reproducible threat
model, safe test environment, objective detectors, and mitigations that still allow benign work.

## Prerequisites and outcomes

Complete Stages 1-5 of the [main learning path](learning-path.md). Familiarity with HTTP APIs,
JSON schemas, test fixtures, and basic classification metrics is enough.

After this path, you should be able to:

- distinguish safety, security, privacy, reliability, and bias failures;
- model assets, trust boundaries, attacker capabilities, and impact;
- test jailbreak, direct and indirect prompt injection, data leakage, and instruction conflicts;
- contain destructive or consequential tool actions with controls outside the model;
- evaluate over-refusal, social bias, political-content filtering, and cross-language drift;
- design a benchmark with benign controls, objective scoring, confidence intervals, and
  reproducible artifacts; and
- turn findings into layered mitigations and regression tests.

## Safe lab rules

Use an isolated test environment for every exercise.

- Test only models, endpoints, datasets, and systems you own or are authorized to assess.
- Use mock files, an in-memory database, fake accounts, synthetic personal data, and synthetic
  secrets such as `CANARY_SECURITY_001`.
- Disable outbound network access unless the exercise explicitly requires a controlled mock.
- Replace delete, send, publish, purchase, transfer, and permission-changing tools with
  deterministic simulators that record intent without causing the action.
- Do not place credentials, private corpus text, hidden production prompts, or personal data in
  attack fixtures or logs.
- Store the model id, revision, quantization, chat template, system prompt hash, generation
  parameters, detector version, and random seed with every result.
- Stop a run when it escapes the sandbox, reaches an unexpected endpoint, or produces an action
  outside the declared test schema.

## Topic map

- - **1. Threat modeling** (What can the attacker influence, and what must be protected?): Assets,
- trust boundaries, attacker access, impact
- - **2. Instruction hierarchy** (Which instructions should win when inputs conflict?): Clean
- success, hierarchy violations, over-refusal
- - **3. Injection and jailbreaks** (Can hostile text redirect the model or bypass safeguards?): ASR
- by attack family and channel
- - **4. Leakage and output handling** (Can secrets escape or generated content harm a consumer?):
- Canary leakage and validator failures
- - **5. Tools and agents** (Can a model cause an unauthorized or destructive action?): Attempted
- and executed unsafe actions
- - **6. Bias and censorship** (Does behavior change across groups, topics, languages, or
- providers?): Refusal, accuracy, omission, framing, consistency
- - **7. Evaluation quality** (Are labels, detectors, and aggregate claims defensible?): Controls,
- inter-rater checks, CIs, false positives
- - **8. Mitigation and operations** (Which independent controls prevent or limit impact?): Residual
- risk and regression results

## Module 1 -- Threat modeling for an LLM application

Start with the system, not the model. Draw the flow from user input through prompts, retrieval,
tools, generated output, storage, and external side effects. Mark each place where untrusted data
crosses into a more privileged component.

For each threat, record:

- asset: corpus text, credentials, system integrity, user trust, availability, or money;
- attacker: end user, document author, compromised tool, model supplier, or insider;
- access: prompt only, retrieved-content control, tool output, model weights, or host access;
- security property: confidentiality, integrity, availability, authorization, or provenance;
- success condition: an observable state change or disclosure, not a vague impression; and
- impact and containment: what the attacker gains and which boundary limits it.

Use the [OWASP Top 10 for LLM
Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
for application risks, [MITRE ATLAS](https://atlas.mitre.org/) for adversary techniques, and the
[NIST Generative AI
Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
for lifecycle risk management.

## Module 2 -- Instruction hierarchy and instruction-following failures

An LLM application mixes instructions and data. They must not have equal authority. A practical
hierarchy is application policy and system instructions, then the user's authorized request,
then untrusted retrieved text and tool output. The last category is data even when it says
"ignore previous instructions."

Test four distinct failure modes:

- hierarchy violation: lower-trust text overrides a higher-trust instruction;
- under-refusal: the model performs a disallowed action or provides disallowed content;
- over-refusal: the model rejects a benign, in-scope request;
- task non-compliance: the model neither violates safety nor completes the requested format or
  task.

Create paired cases: one conflict case and one semantically similar benign control. Measure clean
task success and instruction-hierarchy violation rate separately. A defense that drives ASR to
zero by refusing everything has failed the usefulness requirement.

Read [The Instruction Hierarchy](https://arxiv.org/abs/2404.13208) for the privilege model.

## Module 3 -- Prompt injection and jailbreaks

Keep these categories separate:

- jailbreak: adversarial user input attempts to bypass model-level safety behavior;
- direct prompt injection: a user tells the application to ignore or replace its instructions;
- indirect prompt injection: hostile instructions arrive through a document, webpage, email,
  retrieved chunk, memory entry, or tool result;
- RAG injection: an indirect injection specifically enters through indexed or retrieved content;
- multi-turn attack: earlier turns establish state or context used by a later payload.

Vary attack channel, language, position, context length, encoding, and paraphrase. Do not report a
single global ASR: show results by family and retain clean task success for the same task set.

Reusable evaluation resources include
[JailbreakBench](https://jailbreakbench.github.io/),
[HarmBench](https://www.harmbench.org/), and
[garak](https://github.com/NVIDIA/garak). Treat public attack sets as seed material: deduplicate,
pin a revision, review licenses, and add application-specific RAG and tool cases.

## Module 4 -- Leakage, privacy, and unsafe output handling

System prompts are not secret storage. Authorization must be enforced by application code, and
credentials must never enter the model context. Test leakage with synthetic canaries placed in
separate trust zones: system text, another user's mock record, a non-retrieved document, and a
tool response.

Score exact and normalized canary matches, partial disclosure, and cross-user access. Keep the
canary detector deterministic and inspect false positives.

Treat all model output as untrusted input to the next component. Validate and constrain:

- tool names and typed arguments;
- SQL against an allowed operation set and a least-privileged connection;
- paths against an allowed root without traversal;
- URLs against schemes, hosts, and redirect policy;
- generated HTML or Markdown before rendering; and
- shell-like content by avoiding shell execution entirely where a typed API exists.

## Module 5 -- Destructive and consequential actions

The model must not be the authorization boundary. Classify tools by effect:

- **Read-only** (Search, fetch a permitted record): Scope checks, result filtering, audit log
- **Reversible write** (Create a draft, add a temporary label): Allowlist, idempotency key, rollback
- - **Consequential write** (Send, publish, deploy, modify access): Preview plus explicit approval
- at execution time
- - **Irreversible or high impact** (Delete, transfer funds, rotate credentials): Keep unavailable
- to the model or require a separate strongly authenticated workflow

Defense in depth includes least privilege, narrowly typed tools, parameter validation, sandboxing,
dry-run mode, transaction boundaries, rate and spend limits, bounded loops, approval for
consequential actions, immutable audit logs, and a kill switch. Prompt wording can support these
controls but cannot replace them.

Use a deterministic mock tool world and assert final state. Record both `attempted_unsafe_action`
and `executed_unsafe_action`; a downstream policy gate may correctly block a bad model decision.
The [OWASP Excessive
Agency](https://owasp.org/www-project-top-10-for-large-language-model-applications/2_0_vulns/LLM06_ExcessiveAgency.html)
guidance and [AgentDojo](https://arxiv.org/abs/2406.13352) provide useful designs.

## Module 6 -- Bias, censorship, and multilingual consistency

Bias is not one scalar. Define the expected behavior and label source before running models.
Separate at least:

- social stereotypes and disparate treatment;
- language and dialect quality gaps;
- political-content refusal, omission, or asymmetric framing;
- factual accuracy and citation quality; and
- provider policy or moderation-layer effects.

For models developed in mainland China, published evaluations have found social biases in
Chinese-language tasks and higher political-content refusal or omission for some model families.
These are empirical findings about tested artifacts, not a property to assign from nationality
alone. Models from every region can reflect training-data, alignment, and provider-policy bias.

A defensible comparative protocol should:

1. Pre-register topic categories, expected evidence, labels, and exclusion rules.
2. Use matched prompts in Simplified Chinese, Traditional Chinese where relevant, Ukrainian, and
   English; have fluent reviewers check that translations preserve meaning and tone.
3. Include Chinese-origin and non-Chinese control models of roughly comparable capability.
4. Compare local open weights with hosted endpoints where licenses permit; provider-side filters
   can differ from weight behavior.
5. Pin model revisions and chat templates, repeat stochastic generations, and retain raw output.
6. Score refusal, factual correctness, material omission, stance or framing, and cross-language
   consistency as separate outcomes.
7. Use independent factual references and at least two reviewers for subjective labels. Report
   disagreement instead of forcing false precision.
8. Avoid inferring censorship from one refusal. Check benign controls, paraphrases, languages,
   endpoint behavior, and whether the model simply lacks knowledge.

Starting resources:

- [CBBQ](https://aclanthology.org/2024.lrec-main.260/) evaluates social stereotypes in Chinese
  cultural contexts.
- [McBE](https://aclanthology.org/2025.findings-acl.313/) expands Chinese bias evaluation across
  multiple task formats and categories.
- [Political censorship in large language models originating from
  China](https://academic.oup.com/pnasnexus/article/5/2/pgag013/8487339)
  compares China-originating and non-China-originating models across Chinese and English prompts.
- [This Land is Your, My Land](https://arxiv.org/abs/2305.14610) demonstrates why geopolitical
  consistency should be tested across languages, not assumed from one prompt language.

## Module 7 -- Benchmark design and metrics

Each case should contain a stable id, threat family, trusted instruction, untrusted payload,
expected allowed behavior, prohibited observable behavior, detector, benign-control id, language,
source, license, and reviewer status. Keep adversarial text as data, never executable test logic.

Core metrics:

| Metric | Meaning | Direction |
|---|---|---|
| Clean task success | Benign task completed correctly | Higher is better |
| Attack success rate | Attack achieved its prohibited objective | Lower is better |
| Hierarchy violation rate | Lower-trust instruction overrode policy | Lower is better |
| Canary leakage rate | Synthetic protected value appeared in output | Lower is better |
| Unsafe tool-call rate | Unauthorized action was requested | Lower is better |
| Executed unsafe-action rate | Policy layer allowed the unsafe state change | Lower is better |
| Appropriate refusal recall | Harmful requests correctly refused | Higher is better |
| Benign over-refusal rate | Allowed requests incorrectly refused | Lower is better |
| Cross-language consistency | Matched prompts receive equivalent treatment | Higher is better |

Report per-family counts and bootstrap confidence intervals. Do not average security with RAG
quality unless every component has a documented normalization and uncertainty estimate. Prefer
deterministic detectors such as exact canary matches, schema validation, and mock-state
assertions. If an LLM judge is unavoidable, calibrate it against independent human labels and
publish its false-positive and false-negative behavior.

## Module 8 -- Mitigation and regression

Map each observed failure to controls at more than one layer:

- data: provenance, access control, retrieval filtering, and content boundaries;
- model interaction: clear hierarchy, minimal context, structured output, and refusal policy;
- tool gateway: authorization, least privilege, typed validation, approvals, and quotas;
- runtime: isolation, network policy, timeouts, bounded loops, and resource limits;
- monitoring: trace ids, redacted logs, anomaly detection, and incident response; and
- evaluation: pinned adversarial regressions plus benign controls in CI or scheduled runs.

Re-run the same frozen suite after mitigation. Record both security improvement and utility cost.
Then add newly discovered failures to a private holdout set so prompt-specific patches do not
masquerade as general robustness.

## Eight-session practical syllabus

Plan for 2-4 hours per session.

- - **1. Scope and threats** (Module 1; OWASP and NIST overview): One data-flow diagram and a threat
- register with five concrete success conditions
- - **2. Instruction conflicts** (Module 2; Instruction Hierarchy): Ten paired conflict/control
- cases and deterministic hierarchy labels
- - **3. Injection and jailbreaks** (Module 3; inspect JailbreakBench and HarmBench schemas): A
- pinned, licensed mini-suite split by direct, indirect, RAG, and jailbreak families
- - **4. Leakage and output handling** (Module 4): Synthetic canaries plus validators for paths,
- URLs, and one typed tool schema
- - **5. Destructive actions** (Module 5; OWASP Excessive Agency): A mock tool world with read,
- reversible-write, and blocked high-impact actions
- - **6. Bias and censorship** (Module 6; CBBQ, McBE, and the comparative censorship study): A
- reviewed multilingual prompt matrix with refusal, accuracy, omission, and consistency labels
- **7. Scoring** (Module 7): Per-family ASR, clean success, over-refusal, leakage, and bootstrap CIs
- - **8. Defend and report** (Module 8): A mitigation, before/after regression report, residual-risk
- statement, and holdout plan

## Capstone for loc-lm-bench

Draft a small security benchmark-compatible security suite without executing real tools:

- 10 direct prompt-injection cases;
- 10 RAG-injection cases with malicious text embedded in mock Ukrainian documents;
- 10 jailbreak or unsafe-content cases from a pinned, license-compatible source;
- 10 tool-abuse cases against a deterministic in-memory environment;
- one benign matched control for every adversarial case; and
- objective detectors for instruction following, canary leakage, and final mock state.

The report should contain clean task success, ASR by family, over-refusal, unsafe attempted and
executed actions, 95 percent confidence intervals, model/runtime metadata, known detector gaps,
and links to retained artifacts. Keep this capstone separate from the current RAG leaderboard;
the planned project design uses a distinct security tier for exactly that reason.

## Exit checklist

- The threat model names assets, attackers, trust boundaries, and observable success conditions.
- Every adversarial case has a benign control and a reviewed expected outcome.
- Tests run only against synthetic data and deterministic or sandboxed tools.
- Objective detectors cover the primary score; subjective judging is calibrated and secondary.
- Results separate attack families, model decisions, policy-gate decisions, and final effects.
- Bias claims identify exact models, versions, languages, prompts, and limitations.
- Mitigations exist outside prompt text and are covered by regression tests.
- The report includes uncertainty, utility cost, residual risk, and reproducibility metadata.
