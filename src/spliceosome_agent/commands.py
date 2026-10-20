"""One catalog for interactive help and completion; CLI arguments are separate."""

# Each supported command has its own row, including aliases. Put records first
# so /tasks is visible before the user needs to scroll a small terminal.
COMMON_COMMANDS = (
    ("/tasks", "List runs: status, time and goal"),
    ("/runs", "List runs (alias of /tasks)"),
    ("/plans [prefix]", "List plans or inspect a hash prefix"),
    ("/status [run-id]", "Show run status"),
    ("/report [run-id]", "Show the report"),
    ("/logs [run-id] [n]", "Show log tail (default: 40 lines)"),
    ("/inputs", "Show declared input files"),
    ("/study <file>", "Load and check a study config"),
    ("/demo", "Create and check a synthetic example"),
    ("/dataset <number>", "Choose a dataset (1-based)"),
    ("/scale log2|raw_log2|counts|unknown", "Confirm the measurement scale"),
    ("/swap", "Swap the comparison direction"),
    ("/plan [goal]", "Show the plan; with a goal, prepare it"),
    ("/edit <file>", "Revise the plan from a config file"),
    ("/confirm <hash>", "Confirm this exact plan and run"),
    ("/approve <hash>", "Approve the exact plan; do not run"),
    ("/run", "Run the approved plan"),
    ("/cancel", "Revoke the current approval"),
    ("/stop <run-id>", "Stop a run; keep partial results"),
    ("/resume <run-id>", "Resume with matching fingerprints"),
    ("/models", "Show model config (never keys)"),
    ("/doctor", "Check the environment"),
    ("/help", "Show all available commands"),
    ("/quit", "Exit; background runs continue"),
)


def command_rows(*, tui=False):
    rows = list(COMMON_COMMANDS)
    if tui:
        rows[3:3] = [
            ("/current", "Show the retained task; do not run"),
            ("/home", "Show the welcome screen"),
        ]
        rows = [("/edit [file]", "Open the editor or revise from a file") if usage.startswith("/edit ")
                else (usage, "Configure the model") if usage == "/models"
                else (usage, description) for usage, description in rows]
    return rows


def command_names(*, tui=False):
    return [usage.split()[0] for usage, _ in command_rows(tui=tui)]


def help_text(*, tui=False):
    lines = ["Help", ""]
    lines.extend(f"{usage:<22} {description}" for usage, description in command_rows(tui=tui))
    lines.extend(["", "Optional run-id defaults to the latest run.",
                  "Plain conversation never approves or runs analysis."])
    if tui:
        lines.append("PgUp/PgDn Scroll · Esc Back · F1 Help · Ctrl+Q Quit")
    return "\n".join(lines)
