import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
SURVEY_DIR = DATA_DIR / "survey"
RULES_DIR = DATA_DIR / "rules"
INTERMEDIATE_OUTPUT_DIR = DATA_DIR / "intermediate_output"

SURVEY_A_PATH = SURVEY_DIR / "survey_primary.json"
SURVEY_B_PATH = SURVEY_DIR / "survey_secondary.json"
RULES_PATH = RULES_DIR / "rules_fired.txt"
SNIPPETS_PATH = INTERMEDIATE_OUTPUT_DIR / "snippets.txt"

MODEL_OPTIONS = (
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4o",
    "claude-sonnet-4-5",
    "openrouter/openai/gpt-4o-mini",
    "xai/grok-4-fast-reasoning",
    "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct",
)

MODEL_PROVIDER_BY_NAME = {
    "gpt-4o-mini": "openai",
    "gpt-4.1": "openai",
    "gpt-4o": "openai",
    "claude-sonnet-4-5": "anthropic",
    "openrouter/openai/gpt-4o-mini": "openrouter",
    "xai/grok-4-fast-reasoning": "xai",
    "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct": "huggingface",
}

PROVIDER_ENV_VAR = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
    "huggingface": "HF_TOKEN",
    "bedrock": None,
}


def model_provider(model: str) -> str:
    explicit = MODEL_PROVIDER_BY_NAME.get(model)
    if explicit:
        return explicit
    if model.startswith("openrouter/"):
        return "openrouter"
    if model.startswith("xai/"):
        return "xai"
    if model.startswith("anthropic/") or model.startswith("claude"):
        return "anthropic"
    if model.startswith("huggingface/") or model.startswith("hf/"):
        return "huggingface"
    if model.startswith("bedrock/") or model.startswith("qwen"):
        return "bedrock"
    return "openai"


def required_api_env_var(model: str) -> Optional[str]:
    return PROVIDER_ENV_VAR[model_provider(model)]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_text(path: Path, fallback: str = "") -> str:
    if not path.exists():
        return fallback
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return fallback


class ConnectDemoGUI:
    def __init__(
        self,
        root: tk.Tk,
        selected_model: str,
        provider_api_key: str,
        hf_token: str,
    ) -> None:
        self.root = root
        self.root.title("NARRATE - Chat/Survey/Rules")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 700)
        self.selected_model = selected_model
        self.provider_api_key = provider_api_key
        self.hf_token = hf_token
        self.provider_env_var = required_api_env_var(selected_model)
        self._configure_runtime_environment()
        self.data_a = load_json(SURVEY_A_PATH)
        self.data_b = load_json(SURVEY_B_PATH)
        self.rules_text = load_text(RULES_PATH, "No rules loaded.")
        self.snippets_text = load_text(SNIPPETS_PATH, "No snippets loaded.")
        self._build_ui()

    def _configure_runtime_environment(self) -> None:
        # Keep credentials process-local and available to backend modules.
        if self.provider_env_var and self.provider_api_key:
            os.environ[self.provider_env_var] = self.provider_api_key
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.hf_token

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=2)
        self.root.columnconfigure(1, weight=2)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=10)
        middle = ttk.Frame(self.root, padding=10)
        right = ttk.Frame(self.root, padding=10)
        left.grid(row=0, column=0, sticky="nsew")
        middle.grid(row=0, column=1, sticky="nsew")
        right.grid(row=0, column=2, sticky="nsew")

        self._build_left_column(left)
        self._build_middle_column(middle)
        self._build_right_column(right)

    def _build_left_column(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)

        ttk.Label(parent, text="Input Story").grid(row=0, column=0, sticky="w")
        self.chat_input = tk.Text(parent, height=8, wrap="word")
        self.chat_input.grid(row=1, column=0, sticky="nsew", pady=(2, 8))

        controls = ttk.Frame(parent)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)

        ttk.Label(controls, text="Configured Model").grid(row=0, column=0, padx=(0, 6), sticky="w")
        ttk.Label(controls, text=self.selected_model).grid(row=0, column=1, padx=(0, 12), sticky="w")

        ttk.Label(controls, text="Scenario").grid(row=0, column=2, padx=(0, 6), sticky="e")
        self.scenario_var = tk.StringVar(value="individualistic")
        scenario_dropdown = ttk.OptionMenu(
            controls,
            self.scenario_var,
            "individualistic",
            "individualistic",
            "collectivistic",
        )
        scenario_dropdown.grid(row=0, column=3, sticky="ew")

        send_button = ttk.Button(controls, text="Send", command=self._append_chat)
        send_button.grid(row=0, column=4, padx=(12, 0))

        ttk.Label(parent, text="Reframed Text").grid(row=3, column=0, sticky="sw")
        self.chat_output = tk.Text(parent, wrap="word", state="disabled")
        self.chat_output.grid(row=4, column=0, sticky="nsew")
        parent.rowconfigure(4, weight=1)

    def _build_middle_column(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)

        ttk.Label(parent, text="Add Survey Questions").grid(row=0, column=0, sticky="w")
        self.additional_q = tk.Text(parent, height=8, wrap="word")
        self.additional_q.grid(row=1, column=0, sticky="nsew", pady=(2, 8), padx=(0, 8))

        score_frame = ttk.LabelFrame(parent, text="Original Text Score", padding=10)
        score_frame.grid(row=1, column=1, sticky="nsew", pady=(2, 8))
        score_val = self.data_a.get("original_story", self.data_a.get("survey_score", "N/A"))
        self.primary_score_var = tk.StringVar(value=str(score_val))
        ttk.Label(score_frame, textvariable=self.primary_score_var, font=("TkDefaultFont", 24)).pack(
            anchor="center", expand=True, fill="both"
        )

        questions_frame = ttk.LabelFrame(parent, text="Survey Questions (From JSON)", padding=8)
        questions_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
        questions_frame.columnconfigure(0, weight=1)
        questions_frame.rowconfigure(0, weight=1)

        self.questions_box = tk.Text(questions_frame, wrap="word")
        self.questions_box.grid(row=0, column=0, sticky="nsew")
        q_scroll = ttk.Scrollbar(questions_frame, orient="vertical", command=self.questions_box.yview)
        q_scroll.grid(row=0, column=1, sticky="ns")
        self.questions_box.configure(yscrollcommand=q_scroll.set)
        self._load_questions()

        secondary_score_frame = ttk.LabelFrame(parent, text="Reframed Text Score", padding=10)
        secondary_score_frame.grid(row=3, column=0, columnspan=2, sticky="ew")
        secondary_val = self.data_b.get("transformed_story", self.data_b.get("survey_score", "N/A"))
        self.secondary_score_var = tk.StringVar(value=str(secondary_val))
        ttk.Label(
            secondary_score_frame,
            textvariable=self.secondary_score_var,
            font=("TkDefaultFont", 20),
        ).pack(anchor="center", expand=True, fill="both")

    def _build_right_column(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        rules_frame = ttk.LabelFrame(parent, text="Rules Fired (TXT)", padding=8)
        rules_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        rules_frame.columnconfigure(0, weight=1)
        rules_frame.rowconfigure(0, weight=1)
        rules_box = tk.Text(rules_frame, wrap="word")
        rules_box.grid(row=0, column=0, sticky="nsew")
        rules_box.insert("1.0", self.rules_text)
        rules_box.configure(state="disabled")

        snippets_frame = ttk.LabelFrame(parent, text="Text Snippets (TXT)", padding=8)
        snippets_frame.grid(row=1, column=0, sticky="nsew")
        snippets_frame.columnconfigure(0, weight=1)
        snippets_frame.rowconfigure(0, weight=1)
        snippets_box = tk.Text(snippets_frame, wrap="word")
        snippets_box.grid(row=0, column=0, sticky="nsew")
        snippets_box.insert("1.0", self.snippets_text)
        snippets_box.configure(state="disabled")

    def _append_chat(self) -> None:
        text = self.chat_input.get("1.0", "end").strip()
        if not text:
            return
        msg = (
            f"Model: {self.selected_model} | Scenario: {self.scenario_var.get()}\n"
            f"Provider: {model_provider(self.selected_model)} | "
            f"HF token loaded: {'yes' if bool(self.hf_token) else 'no'}\n"
            f"Input: {text}\n"
            "Response: Placeholder output panel for your backend response.\n\n"
        )
        self.chat_output.configure(state="normal")
        self.chat_output.insert("end", msg)
        self.chat_output.configure(state="disabled")
        self.chat_output.see("end")
        self.chat_input.delete("1.0", "end")

    def _load_questions(self) -> None:
        questions = self.data_a.get("questions", [])
        if not isinstance(questions, list):
            questions = []
        if not questions:
            content = "No survey questions found in JSON."
        else:
            content = "\n\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        self.questions_box.insert("1.0", content)
        self.questions_box.configure(state="disabled")


class StartupConfigDialog:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.result: Optional[dict[str, str]] = None

        self.window = tk.Toplevel(root)
        self.window.title("NARRATE Setup")
        self.window.geometry("640x360")
        self.window.minsize(560, 320)
        self.window.transient(root)
        self.window.grab_set()
        self.window.protocol("WM_DELETE_WINDOW", self._cancel)

        self.model_var = tk.StringVar(value=MODEL_OPTIONS[0])
        self.provider_key_var = tk.StringVar()
        self.hf_token_var = tk.StringVar()
        self.provider_hint_var = tk.StringVar()
        self.provider_name_var = tk.StringVar()

        self._build_ui()
        self._on_model_change()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.window, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        ttk.Label(
            container,
            text="Configure model and credentials before using NARRATE.",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        ttk.Label(container, text="Model").grid(row=1, column=0, sticky="w", pady=(0, 8))
        model_dropdown = ttk.OptionMenu(
            container,
            self.model_var,
            self.model_var.get(),
            *MODEL_OPTIONS,
            command=lambda _v: self._on_model_change(),
        )
        model_dropdown.grid(row=1, column=1, sticky="ew", pady=(0, 8))

        ttk.Label(container, text="Provider API Key").grid(row=2, column=0, sticky="w", pady=(0, 4))
        provider_entry = ttk.Entry(container, textvariable=self.provider_key_var, show="*")
        provider_entry.grid(row=2, column=1, sticky="ew", pady=(0, 4))

        ttk.Label(container, textvariable=self.provider_name_var).grid(
            row=3, column=1, sticky="w"
        )
        ttk.Label(container, textvariable=self.provider_hint_var).grid(
            row=4, column=1, sticky="w", pady=(0, 8)
        )

        ttk.Label(container, text="Hugging Face Token").grid(row=5, column=0, sticky="w", pady=(0, 4))
        hf_entry = ttk.Entry(container, textvariable=self.hf_token_var, show="*")
        hf_entry.grid(row=5, column=1, sticky="ew", pady=(0, 4))
        ttk.Label(
            container,
            text="Used for Hugging Face API calls.",
        ).grid(row=6, column=1, sticky="w")

        button_row = ttk.Frame(container)
        button_row.grid(row=7, column=0, columnspan=2, sticky="e", pady=(16, 0))
        ttk.Button(button_row, text="Cancel", command=self._cancel).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_row, text="Start", command=self._submit).grid(row=0, column=1)

    def _on_model_change(self) -> None:
        provider = model_provider(self.model_var.get())
        env_var = required_api_env_var(self.model_var.get())
        self.provider_name_var.set(f"Provider for selected model: {provider}")
        if env_var is None:
            self.provider_hint_var.set("No direct API key needed for this model.")
        else:
            self.provider_hint_var.set("Used as the provider API key for the selected model.")

    def _cancel(self) -> None:
        self.result = None
        self.window.destroy()

    def _submit(self) -> None:
        model = self.model_var.get().strip()
        provider_key = self.provider_key_var.get().strip()
        hf_token = self.hf_token_var.get().strip()
        env_var = required_api_env_var(model)

        if env_var and not provider_key and env_var != "HF_TOKEN":
            messagebox.showerror("Missing API Key", "Enter a provider API key to continue.")
            return
        if env_var == "HF_TOKEN" and not hf_token and not provider_key:
            messagebox.showerror("Missing HF Token", "Enter a Hugging Face token to continue.")
            return

        self.result = {
            "model": model,
            "provider_api_key": provider_key if env_var != "HF_TOKEN" else (provider_key or hf_token),
            "hf_token": hf_token or (provider_key if env_var == "HF_TOKEN" else ""),
        }
        self.window.destroy()


def main() -> None:
    root = tk.Tk()
    root.withdraw()

    setup = StartupConfigDialog(root)
    root.wait_window(setup.window)
    if not setup.result:
        root.destroy()
        return

    root.deiconify()
    ConnectDemoGUI(
        root,
        selected_model=setup.result["model"],
        provider_api_key=setup.result["provider_api_key"],
        hf_token=setup.result["hf_token"],
    )
    root.mainloop()


if __name__ == "__main__":
    main()
