"""Keyboard-driven selection shared by interactive menus."""
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option


class SelectionCancelled(Exception):
    """The user left a menu without selecting an item."""


class Picker(App):
    CSS = """
    Screen { height: auto; background: #191013; color: #eadde0; }
    #heading { color: #eadde0; text-style: bold; padding: 0 1; background: #191013; }
    OptionList {
        height: auto; max-height: 14; border: round #49383f;
        background: #25191e; color: #e1d2d6;
        scrollbar-color: #66525a; scrollbar-background: #25191e;
    }
    OptionList > .option-list--option { background: #25191e; color: #e1d2d6; }
    OptionList > .option-list--option-highlighted { background: #49383f; color: #fff1f3; text-style: bold; }
    OptionList:focus > .option-list--option-highlighted { background: #49383f; color: #fff1f3; text-style: bold; }
    OptionList > .option-list--option-hover { background: #38292f; color: #fff1f3; }
    #hint { color: #a29196; padding: 0 1; background: #191013; }
    """
    BINDINGS = [Binding("escape", "cancel", "Назад", show=False)]

    def __init__(self, title, options):
        super().__init__()
        self.title_text = title
        self.options = options

    def compose(self) -> ComposeResult:
        yield Label(self.title_text, id="heading", markup=False)
        yield OptionList(*(Option(Text(label), id=str(i)) for i, (label, _) in enumerate(self.options)))
        yield Label("↑/↓ — выбор  ·  Enter — подтвердить  ·  Esc — назад", id="hint")

    def on_mount(self):
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        self.exit(int(event.option.id))

    def action_cancel(self):
        self.exit(None)


def choose(title, options, extra=None):
    entries = list(options)
    if extra:
        entries.append((extra, None))
    if not entries:
        return None
    selected = Picker(title, entries).run(inline=True)
    if selected is None:
        raise SelectionCancelled()
    return entries[selected][1]
