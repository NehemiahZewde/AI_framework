# General Conversational Agent - Phase 2

Phase 1 remains the foundation: this is one general-purpose, ChatGPT-style
conversation agent using the OpenAI Agents SDK and Chainlit. Phase 2 adds
optional OpenAI-native file attachments without turning the app into a file or
machine-learning workflow.

The active application does not use or modify `AI_framework/ai_framework/`.
The earlier dataset wizard is preserved in `archive_wizard_version/`, and the
original chat-only foundation is preserved in `archive_phase1_chat_only/`.

## General Chat And Files

- Ask general questions, follow up naturally, change topics, and return to an
  earlier topic during the current running chat.
- Attach a supported file with the normal Chainlit paperclip icon and ask a
  natural question about it in the same message.
- The attachment is uploaded to the OpenAI Files API with `purpose="user_data"`.
- The app passes the resulting OpenAI file ID to the same agent as an
  `input_file` item alongside the user text.
- When a file is attached without text, the agent briefly acknowledges it and
  asks what the user would like to know.

The Agents SDK `SQLiteSession` keeps conversation history in memory for the
current chat. This includes the file-input turn, allowing natural follow-up
questions without re-uploading the same file. General questions remain normal
conversation before, after, or between file questions.

## Supported Files

- PDF (`.pdf`)
- CSV and TSV (`.csv`, `.tsv`)
- Excel (`.xls`, `.xlsx`)
- Word and PowerPoint (`.docx`, `.pptx`)
- Text and Markdown (`.txt`, `.md`, `.markdown`)
- JSON (`.json`)

The application does not parse files with pandas, PDF, Word, or PowerPoint
libraries in this phase. OpenAI handles file processing. If the Files API or
the model cannot accept a file, the app shows a concise error and the general
chat remains available.

## Attachment Privacy And Lifecycle

Files sent through the paperclip are uploaded to OpenAI for model processing.
The app retains only in-memory metadata for the current Chainlit session: file
ID, filename, MIME type, size, upload time, and status. It does not retain API
keys, file bytes, extracted text, DataFrames, or generated summaries as file
state.

The uploader requests a one-day expiration policy for `user_data` files when
the installed client supports `expires_after`. If the client does not support
that optional field, OpenAI file cleanup is a future improvement. Local
attachment metadata and conversation history disappear when the chat ends or
the Chainlit process stops.

## Files

- `app.py`: API-key flow, paperclip detection, upload coordination, and chat UI.
- `agent.py`: one `gpt-5.5` conversational agent and Responses-format file input.
- `session_manager.py`: in-memory Chainlit-to-Agents conversation mapping.
- `attachment_manager.py`: in-memory attachment metadata per Chainlit chat.
- `openai_file_service.py`: Files API upload and supported-file validation.
- `requirements.txt`: existing pinned runtime dependencies.
- `archive_phase1_chat_only/`: preserved Phase 1 chat-only source.
- `archive_wizard_version/`: preserved earlier guided dataset workflow.

## Run From `ml_main`

From the Neumarker directory:

```powershell
conda activate ml_main
python -m pip install --upgrade -r AI_framework/ml_agent_runner/requirements.txt
python -m chainlit run AI_framework/ml_agent_runner/app.py -w
```

The working pins remain unchanged: `openai-agents==0.18.0`,
`openai==2.36.0`, `pydantic==2.13.4`, `chainlit==2.11.1`, and
`pandas==2.3.3`. pandas remains installed for later phases but is not used by
the active Phase 2 file path. `openai==2.36.0` remains pinned because
`openai==2.45.0` previously caused the Agents SDK `InputTokensDetails` /
`cache_write_tokens` error.

## Manual Test Plan

1. Start the app and enter the API key in the UI.
2. Ask: `What is logistic regression?`
3. Attach a PDF and ask: `Give me a concise summary of this PDF.`
4. Ask: `What was the main conclusion?` without reattaching the PDF.
5. Attach `breast_cancer_coimbra.csv` and ask for its column names, contents,
   and row/column count.
6. Attach one DOCX, PPTX, or XLSX file and ask for a brief explanation.
7. Ask an unrelated general question, then return to a previous file question.
8. Attach an unsupported or malformed file and verify a clear error while chat
   remains usable.
9. Confirm that no ML or dataset workflow starts automatically.

## Known Limitations And Later Phases

- File attachment state and conversation history are in memory only.
- The app does not include Code Interpreter, File Search, vector stores, custom
  parsing, plotting, or detailed code-based table analysis.
- Persistent attachment cleanup and restoration are future improvements.
- `ai_framework` integration, target selection, preprocessing, feature
  selection, and model training remain later phases.
