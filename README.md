# NARRATE

Three-column demo for chat + survey + rules/snippets.

## Run

```bash
pip install streamlit
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Open in browser:

- `http://localhost:8501` (local machine)
- `http://<server-ip>:8501` (remote server)

Optional desktop Tk demo (requires display):

```bash
python3 gui_app.py
```

## Data Layout

- `data/survey/survey_questions.json` (universal across all model/scenario selections)
- `data/rules/<model_key>/<scenario>/rules_fired.txt`
- `data/intermediate_output/<model_key>/<scenario>/snippets.txt`
- `data/output/<model_key>/<scenario>/chat_output.log`
- `output/phase2/<model>/<problem>/<story>/iteration_*/survey.json` (phase outputs used for scoring)
- `data/input/` (reserved for input files)

## Logos

Bottom-right logos are shown by default (Syracuse + ASU).  
To use local logo files instead, place:

- `data/input/logos/syracuse_logo.png`
- `data/input/logos/asu_logo.png`
