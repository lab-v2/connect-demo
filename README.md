# NARRATE

NARRATE: Neurosymbolic Abductive Reasoning for Reframing Texts demonstration. A neurosymbolic framework that uses logical inference to transform text to an audience's view. 
The video of live demonstration of NARRRATE is available at data/demofinal.mp4

## Run
Open-access website: `https://narrate.streamlit.app/`

Or

```bash
pip install streamlit
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

When `streamlit_app.py` starts, it now opens a setup page first:

- Select the model to use.
- Enter the provider API key (OpenAI/Anthropic/OpenRouter/xAI based on model).
- Enter Hugging Face token for Hugging Face API usage.

The app exports these for the current process before running:

- Provider key to the model-specific provider environment variable.
- Hugging Face token to the Hugging Face environment variables.

Open in browser:

- `http://localhost:8501` (local machine)
- `http://<server-ip>:8501` (remote server)

Optional desktop Tk demo (requires display):

```bash
python3 gui_app.py
```

When `gui_app.py` starts, it also opens a setup page first:

- Select the model to use.
- Enter the provider API key (OpenAI/Anthropic/OpenRouter/xAI based on model).
- Enter Hugging Face token for Hugging Face API usage.

The app exports these for the current process before running:

- Provider key to the model-specific provider environment variable.
- Hugging Face token to the Hugging Face environment variables.
