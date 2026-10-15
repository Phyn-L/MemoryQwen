from rich.progress import ProgressColumn
from rich.text import Text


class StepSpeedColumn(ProgressColumn):
    """Show step throughput since this progress task was created."""

    def render(self, task):
        elapsed = task.elapsed
        start_completed = task.fields.get("speed_start_completed", 0)
        if elapsed is None or elapsed <= 0:
            return Text("-- step/s")
        return Text(f"{(task.completed - start_completed) / elapsed:.2f} step/s")
