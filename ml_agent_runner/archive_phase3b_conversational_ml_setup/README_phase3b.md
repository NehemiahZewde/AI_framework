# General Conversational Agent - Phase 3B

Phase 3B keeps the Phase 1 general chat, Phase 2 OpenAI-native attachments,
and Phase 3A local tabular workspace. It adds the first controlled,
conversational standardized dataset setup for a locally attached tabular file.
The application does not import, run, or modify `AI_framework/ai_framework/`.

## What The Agent Can Do

Users can continue to discuss general topics and attached files naturally. For
CSV, TSV, Excel, JSON, and Parquet uploads that load successfully into pandas,
the user can also begin local binary-classification setup conversationally:

1. The user names a target column.
2. The agent validates it and reports its non-null target values.
3. If exactly two values are present, the user chooses the positive class.
4. The agent creates local `df`, `X`, `y`, `feature_names`, `metadata`, and a
   target mapping using float encodings only: negative value to `0.0`, positive
   value to `1.0`.

The agent never guesses a target or positive class. It does not start
train/validation splitting, preprocessing, feature selection, model training,
or `ai_framework` execution. Users may freely switch back to ordinary chat at
any point.

## Three Separate Memory Layers

Each Chainlit chat has three independent, in-memory layers:

1. `SQLiteSession` retains the conversational history used by the OpenAI
   Agents SDK for the active app process.
2. `TabularWorkspace` holds at most one active local pandas DataFrame and its
   compact file metadata.
3. `MLProjectState` holds only the controlled setup objects for that local
   table: selected target, target values, positive and negative classes, float
   target mapping, `df`, `X`, `y`, feature names, metadata, status, and errors.

None of these layers stores the OpenAI API key. The API key remains only in
`cl.user_session`; it is not written to disk, `.env`, JSON, logs, or config.
The local DataFrame is not inserted into model prompts, SDK session history, or
the project state tool responses. The agent sees compact summaries only.

When a later valid tabular attachment replaces the active local table, its
`MLProjectState` is replaced too. Selecting a different target clears the
positive/negative class, target mapping, `X`, `y`, and completed setup.
Changing the positive class clears the mapping and derived setup. All local
state disappears when the chat ends or the Chainlit process stops.

## File Handling

Supported tabular attachments have two independent representations:

1. The original file goes to the OpenAI Files API as a native `input_file` for
   ordinary file-grounded conversation.
2. The local attachment may load into `TabularWorkspace` for controlled pandas
   setup operations.

CSV, TSV, XLSX, XLS, JSON, and Parquet have explicit local loaders. Excel
records every worksheet name and loads the first sheet as the active DataFrame.
PDF, DOCX, PPTX, text, Markdown, and failed table loads remain available for
normal native-file/general chat when their OpenAI upload succeeds; they do not
replace the last valid local table.

## Files

- `app.py`: Chainlit API-key flow, attachments, workspace replacement, and thin
  runtime-context wiring.
- `agent.py`: the `gpt-5.5` general agent and its tool-driven setup guidance.
- `dataset_setup.py`: pure DataFrame validation, safe target-value matching, and
  standardized binary dataset builder.
- `ml_project_state.py`: typed, non-persistent per-chat ML project state.
- `ml_setup_tools.py`: controlled Agents SDK tools for state inspection, target
  selection, positive-class selection, and setup construction.
- `tabular_workspace.py`: typed local pandas workspace and runtime context.
- `tabular_loader.py`: explicit local pandas loader registry.
- `archive_phase3a_local_tabular_workspace/`: preserved Phase 3A source before
  conversational standardized setup was added.

## Run From `ml_main`

From the Neumarker directory, use the confirmed Anaconda environment:

```powershell
conda activate ml_main
python -m pip install --upgrade -r AI_framework/ml_agent_runner/requirements.txt
python -m chainlit run AI_framework/ml_agent_runner/app.py -w
```

Then paste the API key in the Chainlit UI and attach a file with the paperclip.
The pinned SDK combination is `openai-agents==0.18.0`,
`openai==2.36.0`, and `pydantic==2.13.4`; `openai==2.36.0` is intentionally
pinned because newer `openai==2.45.0` produced the Agents SDK
`InputTokensDetails/cache_write_tokens` validation error in this project.

## Manual Test Plan

1. Start a new chat, paste a valid API key, and ask an ordinary question.
2. Attach `breast_cancer_coimbra.csv`; verify the table is loaded locally and
   remains available for normal native-file questions.
3. Say: `Use Classification as the target.` Verify the agent calls the target
   tool, reports `1.0` and `2.0`, and asks which should be positive.
4. Say: `Treat 2 as positive.` Verify the agent builds the setup and reports
   `df` as 116 x 10, `X` as 116 x 9, `y` length 116, all nine biomarker feature
   names, and mapping `1.0 -> 0.0`, `2.0 -> 1.0`.
5. Ask for the current setup and verify it reports only compact metadata and
   shapes, never DataFrame rows or values.
6. Select another target, if available, and verify the positive class, mapping,
   `X`, `y`, and completed status are cleared.
7. Attach a different valid table and verify its local workspace and ML project
   state replace the prior table/setup.
8. Attach a PDF or malformed table and verify the previous valid local setup
   remains while general/native-file conversation is still usable.

## Current Boundary

This phase creates standardized binary dataset objects only. Split settings,
train/validation bundles, preprocessing, feature groups, feature selection,
training, model evaluation, persistence, and arbitrary local code execution
are deliberately out of scope.
