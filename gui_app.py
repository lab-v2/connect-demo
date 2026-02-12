import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk


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
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NARRATE - Chat/Survey/Rules")
        self.root.geometry("1500x900")
        self.root.minsize(1200, 700)
        self.data_a = load_json(SURVEY_A_PATH)
        self.data_b = load_json(SURVEY_B_PATH)
        self.rules_text = load_text(RULES_PATH, "No rules loaded.")
        self.snippets_text = load_text(SNIPPETS_PATH, "No snippets loaded.")
        self._build_ui()

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
        controls.columnconfigure(3, weight=1)

        ttk.Label(controls, text="Model").grid(row=0, column=0, padx=(0, 6), sticky="w")
        self.model_var = tk.StringVar(value="gpt-4o-mini")
        model_dropdown = ttk.OptionMenu(
            controls,
            self.model_var,
            self.model_var.get(),
            "gpt-4o-mini",
            "gpt-4.1",
            "llama-3.1",
            "mistral-large",
        )
        model_dropdown.grid(row=0, column=1, padx=(0, 12), sticky="ew")

        ttk.Label(controls, text="Scenario").grid(row=0, column=2, padx=(0, 6), sticky="w")
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

        ttk.Label(parent, text="Transformed Story").grid(row=3, column=0, sticky="sw")
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

        score_frame = ttk.LabelFrame(parent, text="Original Story Score", padding=10)
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

        secondary_score_frame = ttk.LabelFrame(parent, text="Transformed Story Score", padding=10)
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
            f"Model: {self.model_var.get()} | Scenario: {self.scenario_var.get()}\n"
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


def main() -> None:
    root = tk.Tk()
    ConnectDemoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
