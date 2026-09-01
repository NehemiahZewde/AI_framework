# General Conversational Agent - Dataset Setup And Validation Preparation

The application remains one general-purpose Chainlit chat. General questions,
multi-turn conversation, OpenAI-native file understanding, and paperclip
attachments are still the default behavior. An optional structured ML
preparation workflow starts only when the user asks to prepare the active
dataset for the ML framework.

The application does not modify `AI_framework/ai_framework/`. It imports the
existing preparation module only after the user confirms execution.

## Let's Define What The Model Should Predict

Users can request ML preparation conversationally, for example: `Help me
prepare this dataset for modeling.` The agent inspects only the active table and
returns one combined proposal containing the likely target, its two usable
values, any evidence-backed class meanings, the proposed positive outcome, and
the float mapping.

For an unambiguous binary dataset, the normal interaction is:

1. The agent proposes the complete prediction-target setup.
2. The user replies `Continue` or gives corrections in one message.
3. The deterministic confirmation tool creates and stores `df`, `X`, `y`,
   `feature_names`, `metadata`, and `target_mapping` automatically.

There is no separate classification question, binary-classification question,
positive-class question, or final create question unless information is truly
missing. Multiple plausible target columns require a target choice. Numeric
class labels with undocumented meanings require the user to identify the
positive value. No train/validation split, preprocessing, feature selection, or
model training runs during this stage.

## Prepare The Training And Validation Data

After the initial dataset setup is complete, the user can explicitly continue
to the next stage. The agent presents one primary settings table with the
recommended internal split, 20% final validation, random seed 42, and
stratification enabled. Advanced operational settings are summarized in one
sentence by default. Their detailed table, including the internal outcome name,
appears only when explicitly requested or when an advanced value is changed.
Recommended values are emphasized with Markdown bold text, so no custom colors
or CSS are required.

The interaction is deliberately short:

1. Review one configuration screen.
2. Accept the recommendations or change any subset in natural language.
3. Review one consolidated Step 1 audit table and the exact resolved call.
4. Confirm once, then run the framework function.

Examples such as `Use 25% validation`, `Use random seed 100 and disable
stratification`, and `Keep everything else the same` preserve every unmentioned
setting. All requested changes are validated together before shared state is
updated. Invalid values leave the previous valid configuration intact.

The unchanged framework genuinely supports three approaches:

- Internal split: creates training and final-validation groups from the active
  dataset.
- Separate validation dataset: holds the next supported tabular upload apart
  from the active training dataset, checks target values and shared features,
  then passes the supported external inputs to the framework.
- No separate final-validation set: keeps every active row available for model
  development and returns no final-validation object. This is represented by
  `validation_size=0.0`, which the framework reports as `train_only`.

Choosing no separate final-validation set does not remove future evaluation.
Cross-validation, nested cross-validation, bootstrapping, or another supported
resampling strategy can be configured later during model development. This can
be useful when holding out observations would substantially reduce an already
small training sample.

The pre-execution checkpoint is titled `Step 1 — Review raw data preparation`.
It combines dataset shapes, target mapping, validation decisions, reporting
settings, and a provenance column in one table. The Source column distinguishes
uploaded or metadata-derived inputs, framework defaults, accepted agent
recommendations, explicit user selections, user confirmations, and derived
values.

The exact `mdp.prepare_train_validation_bundles(...)` call appears directly
below the table. Large objects remain represented by `X`, `y`, `feature_names`,
`metadata`, and external-validation variable names. The review is generated
from structured state and is never reconstructed from earlier chat text.

Each review stores `reviewed_prepare_bundles_config`,
`step_1_review_status`, and an incrementing `step_1_review_version`. A setting
change regenerates both the table and call before asking `Run Step 1 using this
configuration?` again. Merely displaying the review does not create or replace
training, validation, or preparation-result objects.

After the single final confirmation in a later user turn, the runner verifies
that the reviewed configuration fingerprint still matches the current
structured state, then calls the real
`mdp.prepare_train_validation_bundles(...)` function. The fingerprint covers
the settings, target mapping, active data-object identities, feature names,
metadata identity, and external-validation inputs without hashing or exposing
row values. If anything changed after review, execution is blocked and an
updated review must be confirmed.

Framework stdout is captured because the framework's structured
`prep_meta["progress_log"]` does not contain the full header, section markers,
separators, and final pipeline line. The captured native output is displayed
once in a `text` code block, followed only by:

`Step 1 completed successfully. The returned features remain raw; no preprocessing has run.`

The runner stores `train_bundle`, the optional `validation_bundle`,
`prep_meta`, the execution log, the executed review version and fingerprint,
the resolved configuration, and an execution timestamp in the in-memory
`MLProjectState`. New outputs replace old ones only after a successful run. A
failed run stores its partial log and error state while preserving any prior
valid bundles. A plain duplicate run is rejected; `Rerun Step 1` is required
to execute again.

`Show me the Step 1 execution log` returns the saved log directly without
running the framework. Exact result questions use compact structured bundle
facts rather than parsing the displayed log. Full bundles and DataFrames are
never placed in the model prompt or visible chat output.

The evidence hierarchy is explicit metadata or documentation, then user
statements, then cautious semantic inference, then unknown. Self-describing
strings such as `control` and `disease` can support a proposal, but numeric
labels alone never establish class meaning. User corrections always override a
proposal. The original negative value maps to float `0.0`, and the original
positive value maps to float `1.0`; the original target values remain unchanged
in `df` and `y`.

## Interruptions And Resumption

ML preparation is not a Chainlit wizard or an app-level message router. The
agent uses deterministic tools and an in-memory workflow state, while normal
conversation stays available. A user can ask an unrelated question while a
decision is pending, then say `Where were we?`, `Continue`, or `What should I
do next?` to resume from the recorded step.

The workflow can also be cancelled. Cancellation clears target/task/class/setup
decisions but retains the active uploaded dataset and general conversation.

## Memory And Privacy

Each chat has three separate in-memory layers:

1. `SQLiteSession` holds Agents SDK conversation history for the running app.
2. `TabularWorkspace` holds one active local pandas DataFrame and compact file
   metadata.
3. `MLProjectState` holds dataset setup objects plus structured target evidence,
   confidence, pending status, and the proposed class orientation. It also holds
   the training/validation workflow state, optional external validation table, and
   returned framework bundles.

Conversation history is not the workflow state. The workflow state is not
saved to disk, and the DataFrame is not placed in model prompts or tool
responses. The OpenAI API key remains only in `cl.user_session`, never in
files, environment configuration, JSON, or logs.

Replacing the active tabular dataset resets the project and workflow state.
Target corrections are validated against the same shared state before the
derived setup is replaced.

## Workflow Tools

- `start_prediction_target_setup()`: creates one evidence-aware target proposal
  or asks for a target choice when candidates are genuinely ambiguous.
- `revise_prediction_target_proposal(...)`: records a target selection or
  partial documented correction and returns the one remaining decision.
- `confirm_prediction_target_setup(...)`: confirms or corrects the complete
  proposal and atomically builds all dataset setup objects.
- `get_prediction_target_status()`: reports the saved proposal, evidence
  sources, pending decision, or completion summary.
- `inspect_target_candidates()`: returns compact candidate evidence including
  dtype, missing count, distinct-value count, representative values, and
  outcome-like name hints without exposing table rows.
- `cancel_ml_preparation()`: clears ML decisions while retaining the dataset.
- `get_standardized_dataset_setup()`: reports the compact stored setup summary
  after it is built.
- `start_prepare_bundles_stage()`: starts or resumes preparation and displays
  the recommended internal configuration in one screen.
- `set_prepare_bundles_validation_mode(validation_mode: str)`: selects an
  internal split, a separate validation dataset, or no separate final-
  validation set.
- `update_internal_prepare_bundles(...)`: validates and applies partial grouped
  updates while retaining every unmentioned setting.
- `configure_external_prepare_bundles(...)`: resolves uploaded external X/y
  and validates its grouped operational settings.
- `get_prepare_bundles_status()`: reports compact configuration, review, or
  result status without full stored objects.
- `show_prepare_bundles_advanced_settings()`: displays the detailed operational
  settings without changing workflow state.
- `inspect_prepare_bundles_function_call()`: repeats the stored reviewed call
  when the user explicitly asks for it again.
- `show_step_1_execution_log()`: returns the saved native framework log without
  rerunning or changing state.
- `inspect_step_1_results()`: returns compact machine-readable settings and
  result facts for conversational answers without parsing the log.
- `run_prepare_train_validation_bundles(allow_rerun: bool)`: runs the real
  framework function only after review and explicit confirmation; `true` is
  reserved for an explicit rerun.

Every model-visible tool argument has an explicit supported type. The startup
schema guard checks each registered function tool before `Runner.run`.

## Current Boundary

The workflow stops after training/final-validation data objects are created. It does
not run feature-group inference, raw feature cleaning, high-cardinality
handling, categorical encoding, imputation, scaling, feature-name sanitization,
preprocessing QC, `preprocess_train_validation_bundles`, feature selection,
model training, evaluation, or persistent bundle storage across app restarts.

## Streaming And Local Performance Logs

Assistant text now streams incrementally into one Chainlit message by using
`Runner.run_streamed(...)` and Chainlit's `stream_token(...)`. Tool execution
can pause visible text temporarily; the same message resumes when model text
continues and is finalized once at the end of the turn.

Deterministic workflow tools use the Agents SDK `tool_use_behavior` callback.
Their validated structured result is rendered as complete Markdown and becomes
the turn's final output, so the model is not called again merely to restate a
known transition. `inspect_target_candidates` remains a model-synthesis tool:
its evidence returns to the same conversational agent for interpretation.

Direct tool output is emitted as one completed chunk through the same Chainlit
message used for streaming, so there is no empty or duplicate assistant
message. The Agents SDK still records the model tool call and structured tool
result in `SQLiteSession`; later turns also read the canonical in-memory
`MLProjectState`. The rendered Markdown is presentation only and is never
parsed to recover state.

Development timing is enabled by default and printed only in the local
terminal. Each turn receives a short correlation ID. The logs capture:

- attachment access, OpenAI file upload, and local pandas loading time
- Agents SDK start, model round trips, and first streamed model event
- time to the first text token successfully sent to Chainlit
- tool names, statuses, individual durations, count, and total tool time
- whether a tool result was displayed directly and whether a second model
  synthesis turn ran
- final model output, final Chainlit completion, and total turn duration
- whether an attachment was processed and native OpenAI file input was used

Timing logs do not include API keys, user messages, model responses,
filenames, file contents, raw bytes, DataFrame contents, tool arguments, or
row-level values. Existing exception logging continues to redact the session
API key.

Training/validation state transitions also emit a compact `[ML_STATE]` line containing
the Chainlit session ID, Python state-object ID, tool name, validation mode
before/after, and workflow status before/after. This verifies that consecutive
turns use the same in-memory project state without logging user text or data.

The canonical stored validation modes are `internal`, `external`, and `none`.
Natural input aliases are normalized before state mutation.

To disable timing output for a PowerShell process before starting Chainlit:

```powershell
$env:ML_AGENT_PERF_LOGGING = "0"
python -m chainlit run AI_framework/ml_agent_runner/app.py -w
```

Remove that environment variable or set it to `1` to enable timing again.
Model switching is intentionally outside this performance step; the agent
continues to use `gpt-5.5`.

### Direct-output Classification

The deterministic direct-output tools are:

- `get_active_tabular_workspace`
- `start_prediction_target_setup`
- `revise_prediction_target_proposal`
- `confirm_prediction_target_setup`
- `get_prediction_target_status`
- `cancel_ml_preparation`
- `get_standardized_dataset_setup`
- `start_prepare_bundles_stage`
- `set_prepare_bundles_validation_mode`
- `update_internal_prepare_bundles`
- `configure_external_prepare_bundles`
- `get_prepare_bundles_status`
- `show_prepare_bundles_advanced_settings`
- `show_step_1_execution_log`
- `run_prepare_train_validation_bundles`

`inspect_target_candidates`, `inspect_prepare_bundles_function_call`, and
`inspect_step_1_results` remain model-synthesis tools because their technical
results need conversational interpretation. General questions that need no
tool continue as ordinary single-model streamed turns.

### Controlled Performance Tests

Run each case three times and use the middle value after sorting the three
results as the median. Record `first_visible_token`, `total`, `tool_turns`,
`tool_total`, and `model_turns` from each `[PERF ...] summary` line.

1. Ask `Explain logistic regression in five sentences.` Expect one streamed
   model turn and no tool.
2. Ask `Prepare this dataset for modeling.` Expect one combined target proposal,
   one tool, and direct output.
3. For undocumented numeric labels, say `Use 2.0 as the positive outcome.`
   Expect the setup objects to be created immediately and reported directly.
4. For an evidence-backed complete proposal, say `Continue.` Expect automatic
   object creation with no separate task-type or create confirmation.
5. Before confirmation, ask why the target was proposed. The analytical
   inspection tool may use a second model synthesis turn while preserving the
   pending proposal.

For deterministic workflow turns, the desired summary fields are
`model_turns=1`, `tool_turns=1`, `direct_tool_output=true`, and
`second_model_synthesis=false`. Analytical tool turns may report
`model_turns=2`, `direct_tool_output=false`, and
`second_model_synthesis=true`. Upload turns still separate Files API latency
from local pandas and model latency.

## Files

- `app.py`: Chainlit UI, API-key flow, attachment handling, and thin runtime
  wiring.
- `agent.py`: the `gpt-5.5` general agent, instructions, tools, and schema
  validation.
- `dataset_setup.py`: pure DataFrame setup helper and target-value resolver.
- `target_setup_workflow.py`: evidence-aware target proposal state, generic
  semantic inference, compact status, and polished direct Markdown.
- `target_setup_tools.py`: deterministic proposal, revision, confirmation, and
  status tools for the condensed first modeling stage.
- `ml_project_state.py`: in-memory dataset objects and structured workflow
  state/transitions.
- `ml_setup_tools.py`: shared compact setup inspection and cancellation tools,
  plus retained compatibility helpers that are not registered in the agent.
- `tabular_workspace.py`: active local dataset and runtime context.
- `tabular_loader.py`: explicit loader registry for tabular attachments.
- `archive_phase3b_conversational_ml_setup/`: Phase 3B snapshot taken before
  this workflow was added.
- `performance.py`: privacy-safe monotonic turn, model, tool, file, and UI
  timing.
- `direct_tool_output.py`: selective Agents SDK stop policy and deterministic
  user-facing Markdown rendering.
- `prepare_bundles_workflow.py`: training/validation workflow state, defaults,
  validation, compact summaries, resolved-call rendering, and Markdown.
- `prepare_bundles_tools.py`: deterministic configuration and framework
  execution tools.

## Run From `ml_main`

From the Neumarker directory:

```powershell
conda activate ml_main
python -m pip install --upgrade -r AI_framework/ml_agent_runner/requirements.txt
python -m chainlit run AI_framework/ml_agent_runner/app.py -w
```

Then paste the API key into the Chainlit UI and attach a file with the
paperclip. Dependency pins remain unchanged, including
`openai-agents==0.18.0`, `openai==2.36.0`, and `pydantic==2.13.4`.

Run the state-transition regression test without launching Chainlit:

```powershell
cd AI_framework/ml_agent_runner
python -m unittest discover -s tests -v
```

## Manual Test Sequence

1. Start the app, enter an API key, and ask a general question.
2. Attach `breast_cancer_coimbra.csv`, ask for its column names, and verify no
   ML workflow starts automatically.
3. Say: `Prepare this dataset for modeling.` Verify `Classification` is proposed
   and values `1.0` and `2.0` are shown, but no class meaning is invented for a
   generic upload.
4. Say: `2.0 is the positive outcome.` Verify the target is confirmed and the
   six setup objects are created immediately with `1.0 -> 0.0` and
   `2.0 -> 1.0`. There must be no separate create question.
5. Ask: `Show me the dataset setup.` Verify compact shapes, feature names,
   metadata, and mapping are shown without DataFrame rows.
6. In a fresh session, use a dataset with `control` and `disease` labels. Verify
   the agent proposes control as negative and disease as positive, explains
   that the labels suggest this interpretation, and asks for one confirmation.
7. Reply `Continue.` Verify automatic completion. Repeat and correct it with
   `Use control as positive`; verify the user correction reverses the mapping.
8. Use a dataset with two plausible outcome columns. Verify the agent asks which
   one to predict and inspects only the selected column afterward.
9. Use a three-value target. Verify the binary-only limitation is shown and no
   mapping or setup objects are created.
10. In a fresh session, begin preparation and cancel it. Verify the dataset
    remains available and the ML decisions are cleared.
11. From a completed target setup, say: `Continue to the next stage.` Verify
    one primary configuration table appears with bold recommendations, a
    compact advanced-settings sentence, and natural-language examples.
12. Say: `Use the recommended settings.` Verify the Step 1 audit shows all
    dataset, target, validation, and reporting inputs with sources. Confirm the
    exact call contains validation size `0.20`, random state `42`,
    stratification enabled, float target encodings, and no full data contents.
    Verify it asks `Run Step 1 using this configuration?` without executing.
13. Say: `Show the advanced settings.` Verify the detailed advanced table
    appears and the pending configuration does not change.
14. Before execution, ask: `What does stratification do?` Verify the agent
    answers normally and the final-confirmation state remains pending.
15. Say: `Actually, use 25% validation and random state 100. Keep everything
    else the same.` Verify both the audit table and exact call regenerate, the
    review version increases, and no preparation results are created.
16. Say: `Use validation size 1.5.` Verify a clear error and that the prior
    valid configuration remains unchanged.
17. Restore 20% validation, use random seed `0`, review the audit, then say in
    a separate turn: `Yes, prepare the data.` Verify one native framework log
    block appears with its `[OK]` and `[SKIP]` lines, followed by exactly the
    concise Step 1 completion statement. Verify there is no result table and
    Step 2 does not begin.
18. Ask: `Was stratification enabled and how many validation rows were
    created?` Verify the answer uses structured result state and does not rerun
    Step 1. Then ask: `Show me the Step 1 execution log.` Verify the saved log
    appears without creating new bundles.
19. Say: `Run Step 1.` Verify no rerun occurs and the app requests explicit
    rerun wording. Then say: `Rerun Step 1.` Verify replacement occurs only
    after success. Ask: `Show me the exact framework call.` Verify the stored
    reviewed call is repeated without changing configuration.
20. To test external validation in a fresh session, select external validation,
    attach a separate tabular validation file, confirm its target column, and
    verify the original active training dataset is not replaced.
21. In another fresh session, say: `Use all current data without a separate
    final-validation set.` Review and confirm the selection. Verify all rows
    remain available for model development, no final-validation object is
    created, and later resampling-based evaluation is explained neutrally.
