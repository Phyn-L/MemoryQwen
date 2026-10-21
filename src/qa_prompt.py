"""Question formatting shared by memory QA and the ICL baseline."""


def format_question(question: str, dataset: str, metadata: dict) -> str:
    options = metadata.get("options") if dataset == "race" else None
    if not options:
        return question
    choices = "\n".join(str(option).strip() for option in options)
    return f"{question}\nOptions:\n{choices}"
