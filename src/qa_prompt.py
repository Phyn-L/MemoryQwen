"""Question formatting shared by training and memory-model evaluation."""


def format_question(question: str, dataset: str, metadata: dict) -> str:
    options = metadata.get("options") if dataset == "race" else None
    if not options:
        return question
    choices = "\n".join(str(option).strip() for option in options)
    return f"{question}\nOptions:\n{choices}"
