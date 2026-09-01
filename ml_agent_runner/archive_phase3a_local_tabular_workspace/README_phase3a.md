# General Conversational Agent - Phase 3A

The Phase 1 general chat and Phase 2 OpenAI-native file understanding remain
the foundation. Phase 3A adds a separate, local pandas workspace for supported
tabular uploads, creating a bridge for later ML-framework work without starting
an ML workflow now.

The active application does not use or modify `AI_framework/ai_framework/`.
Earlier versions remain preserved in `archive_wizard_version/`,
`archive_phase1_chat_only/`, and `archive_phase2_native_files/`.

## Two File Representations

When a supported tabular file is attached through the Chainlit paperclip, the
app keeps two independent representations:

1. The file is uploaded to the OpenAI Files API and sent to the conversational
   agent as `input_file`, for natural document and file questions.
2. The same local attachment is loaded into one in-memory pandas
   `TabularWorkspace`, reserved for future ML-framework operations.

The DataFrame is never injected into the model prompt or Agents SDK session
history. The agent can obtain compact workspace facts only by calling
`get_active_tabular_workspace()`. No target selection, preprocessing, feature
selection, model training, or `ai_framework` execution starts automatically.

Non-tabular files such as PDFs, DOCX files, PPTX files, text, and Markdown
continue through OpenAI-native file understanding only. They do not clear the
existing active local table.

## Local Tabular Formats

- CSV (`.csv`) through `pandas.read_csv`
- TSV (`.tsv`) through `pandas.read_csv(..., sep="\t")`
- Excel (`.xlsx`) through `openpyxl`
- Excel (`.xls`) through `xlrd`
- JSON (`.json`) through `pandas.read_json`
- Parquet (`.parquet`) through `pyarrow`

For Excel files, the loader records all worksheet names and loads only the
first worksheet as the active local DataFrame. A later valid tabular upload
replaces the active local table. A failed local load preserves the previously
valid workspace and does not interrupt OpenAI-native file conversation.

## Memory And Privacy

Each Chainlit chat has one in-memory Agents SDK `SQLiteSession`, attachment
metadata for OpenAI file IDs, and at most one local `TabularWorkspace`. The
workspace stores the DataFrame and compact metadata only in application memory;
it contains no API key and disappears when the chat ends or the app stops.

Files attached in the UI are uploaded to OpenAI for model processing. The app
retains only OpenAI file metadata locally, not file bytes or extracted text.
The uploader requests a one-day `user_data` expiration policy when supported.

## Files

- `app.py`: API-key flow, paperclip processing, dual upload/loading, and chat UI.
- `agent.py`: one `gpt-5.5` agent, native `input_file` input, and workspace tool.
- `session_manager.py`: in-memory Chainlit-to-Agents conversation mapping.
- `attachment_manager.py`: in-memory OpenAI file metadata per chat.
- `openai_file_service.py`: OpenAI Files API upload service.
- `tabular_workspace.py`: typed per-chat local pandas workspace and runtime context.
- `tabular_loader.py`: explicit local tabular registry and pandas loader.
- `archive_phase2_native_files/`: preserved Phase 2 native-file implementation.

## Run From `ml_main`

From the Neumarker directory:

```powershell
conda activate ml_main
python -m pip install --upgrade -r AI_framework/ml_agent_runner/requirements.txt
python -m chainlit run AI_framework/ml_agent_runner/app.py -w
```

Core pins are unchanged: `openai-agents==0.18.0`, `openai==2.36.0`,
`pydantic==2.13.4`, `chainlit==2.11.1`, and `pandas==2.3.3`. Local workbook and
Parquet support adds `openpyxl==3.1.5`, `xlrd==2.0.2`, and `pyarrow==21.0.0`.
If an optional engine is absent, the file can still be discussed through the
OpenAI-native path when its upload succeeds; the app reports the local pandas
preparation failure clearly.

## Manual Test Plan

1. Start the app, enter the API key, and ask a normal general question.
2. Attach `breast_cancer_coimbra.csv` and ask about its contents.
3. Ask: `Is this table also prepared locally for future ML-framework operations?`
4. Verify the agent calls the workspace tool and reports 116 rows, 10 columns,
   and the local column names.
5. Attach an XLSX workbook and verify the first sheet, available sheets, shape,
   and columns are reported.
6. Attach a second valid table and verify it replaces the active local table.
7. Attach a PDF, ask for a summary, and verify the active local table remains.
8. Attach a malformed tabular file and verify the local error is clear, the
   prior local table remains active, and native/general chat stays usable.
9. Confirm that no target-selection or ML workflow begins automatically.

## Current Limits And Next Phase

- One active local DataFrame is supported per chat; there is no persistent
  workspace storage or worksheet-selection interface yet.
- No detailed local statistics, plotting, arbitrary code execution, or
  `ai_framework` execution tools are included.
- The next phase can add controlled, privileged ML-framework operations on the
  local workspace after explicit user direction.
