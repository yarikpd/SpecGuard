"""Keyboard-driven selection shared by interactive menus."""
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input, Label, OptionList
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
    BINDINGS = [Binding("escape", "cancel", "Назад", show=False), Binding("f", "search", "Поиск", show=False), Binding("d", "delete", "Удалить", show=False)]

    def __init__(self, title, options, searchable=False, deletable=(), on_delete=None):
        super().__init__()
        self.title_text = title
        self.options = options
        self.searchable = searchable
        self.deletable = deletable
        self.on_delete = on_delete
        self.deleted = set()

    def compose(self) -> ComposeResult:
        yield Label(self.title_text, id="heading", markup=False)
        if self.searchable:
            yield Input(placeholder="Поиск в текущей папке…", id="search")
        yield OptionList(*(Option(Text(label), id=str(i)) for i, (label, _) in enumerate(self.options)))
        yield Label("↑/↓ — выбор  ·  Enter — подтвердить  ·  Esc — назад" + ("  ·  F — поиск" if self.searchable else "") + ("  ·  D — удалить сохранённый репозиторий" if self.on_delete else ""), id="hint")

    def on_mount(self):
        if self.searchable:
            self.query_one(Input).display = False
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        self.exit(int(event.option.id))

    def action_delete(self):
        listing = self.query_one(OptionList)
        if not self.on_delete or listing.highlighted is None:
            return
        index = int(listing.get_option_at_index(listing.highlighted).id)
        value = self.options[index][1]
        if value in self.deletable:
            self.on_delete(value)
            self.deleted.add(index)
            listing.remove_option(str(index))

    def action_search(self):
        if self.searchable:
            field = self.query_one(Input)
            field.display = True
            field.focus()

    def on_input_changed(self, event: Input.Changed):
        listing = self.query_one(OptionList)
        listing.clear_options()
        query = event.value.casefold()
        listing.add_options(Option(Text(label), id=str(i)) for i, (label, _) in enumerate(self.options)
                            if i not in self.deleted and (query in label.casefold() or label == "📁 .."))
        if listing.option_count:
            listing.highlighted = 0

    def on_input_submitted(self, event: Input.Submitted):
        self.query_one(OptionList).focus()

    def action_cancel(self):
        if self.searchable and self.query_one(Input).display:
            field = self.query_one(Input)
            field.value = ""
            field.display = False
            self.query_one(OptionList).focus()
        else:
            self.exit(None)


def choose(title, options, extra=None, searchable=False, deletable=(), on_delete=None):
    entries = list(options)
    if extra:
        entries.append((extra, None))
    if not entries:
        return None
    selected = Picker(title, entries, searchable=searchable, deletable=deletable, on_delete=on_delete).run(inline=True)
    if selected is None:
        raise SelectionCancelled()
    return entries[selected][1]
