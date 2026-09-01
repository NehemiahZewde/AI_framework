# Dataset Explorer Agent

This is the first source-only agent runner for `AI_framework`. It does not
modify or invoke the existing reusable `ai_framework` modules.

Version 1 scope:

- User uploads one CSV file through Chainlit.
- pandas loads and profiles the CSV.
- The app reports rows, columns, column names, dtypes, missingness, unique
  counts, likely numeric/categorical columns, possible target columns, and
  possible ID-like columns.
- The app asks the user to confirm the target column.
- The app asks whether this is binary classification.
- The app summarizes the draft ML setup with target, task type, feature
  columns, and columns to drop.

Out of scope:

- Model training.
- Feature selection.
- Changes to existing `ai_framework` modules.

## Files

- `app.py`: Chainlit upload and chat flow.
- `agent.py`: OpenAI Agents SDK runtime wrapper.
- `tools.py`: Agents SDK tool for profiling an uploaded CSV.
- `dataset_profile.py`: pandas CSV loading, profiling, heuristics, and Markdown rendering.
- `state.py`: Chainlit session-state keys and state container.
- `requirements.txt`: Dependencies for the runner.

## Run From `ml_main`

Activate `ml_main` first. Then, from the Neumarker directory:

```powershell
conda activate ml_main
python -m pip install --upgrade -r AI_framework/ml_agent_runner/requirements.txt
python -m chainlit run AI_framework/ml_agent_runner/app.py -w
```

This runner pins `openai==2.36.0` because `openai==2.45.0` caused an OpenAI
Agents SDK response-usage parsing error:

```text
ValidationError: InputTokensDetails
cache_write_tokens Field required
```

Optional model override:

```powershell
$env:ML_AGENT_MODEL = "gpt-5.5"
```

When the Chainlit app opens:

1. Paste your OpenAI API key in the UI.
2. Wait for the key validation message.
3. Upload one CSV file.

The API key is kept only in Chainlit session state for the current app session.
It is not written to disk, `.env`, JSON, or config files. When the process stops
or a new session starts, the app asks for the key again.

`OPENAI_API_KEY` can still be used as a developer fallback if you instantiate
`DatasetExplorerAgent` outside the Chainlit flow, but the app itself asks for a
key in the UI first.
