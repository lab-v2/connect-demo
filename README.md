# NARRATE

NARRATE: Neurosymbolic Abductive Reasoning for Reframing Texts demonstration.

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

- `data/input/`
- `data/input/input.txt` (default active story)
- `data/input/<story_name>.txt` (uploaded/selected stories)
- `data/input/logos/` (optional local logo assets, if used)
- `data/individualistic_questions.json`
- `data/collectivistic_questions.json`
- `data/survey/individualistic/survey_questions.json`
- `data/survey/individualistic/additional_survey.json`
- `data/survey/collectivistic/survey_questions.json`
- `data/survey/collectivistic/additional_survey.json`
- `data/rules/<model_key>/<scenario>/pyreason_rules.txt`
- `data/rules/<model_key>/<scenario>/selected_rules.txt`
- `output/phase2/<model>/<problem>/<story>/iteration_<n>/story.txt`
- `output/phase2/<model>/<problem>/<story>/iteration_<n>/survey.json`
- `output/phase2/<model>/<problem>/<story>/iteration_<n>/ground_atoms.json`
- `output/phase2/<model>/<problem>/<story>/iteration_<n>/segments_metadata.json`
- `output/phase2/<model>/<problem>/<story>/iteration_<n>/ranked_prescriptions.json`
- `output/phase2/<model>/<problem>/<story>/iteration_<n>/story_transformed.txt`
- `output/phase2/<model>/<problem>/<story>/iteration_<n>/transformation_log.json`
