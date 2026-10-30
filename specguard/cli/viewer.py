"""Scrollable report viewer that does not depend on terminal scrollback."""
import re
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Label, Static


class ReportViewer(App):
    CSS = """
    Screen { background: #191013; color: #eadde0; }
    #report { height: 1fr; scrollbar-color: #725d65; scrollbar-background: #25191e; }
    #content { height: auto; padding: 0 1; }
    #help { height: auto; padding: 1; background: #38292f; color: #eadde0; }
    """
    BINDINGS = [
        Binding("escape,q", "close", "Назад"),
        Binding("up", "move('up')", show=False),
        Binding("down", "move('down')", show=False),
        Binding("pageup", "move('page_up')", show=False),
        Binding("pagedown,space", "move('page_down')", show=False),
        Binding("home", "move('home')", show=False),
        Binding("end", "move('end')", show=False),
    ]

    def __init__(self, report):
        super().__init__()
        self.report = report

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="report"):
            yield Static(self.report, id="content")
        yield Label("↑/↓ · PgUp/PgDn · колесо — прокрутка | Home/End — начало/конец | Esc — назад", id="help")

    def on_mount(self):
        self.query_one("#report").focus()

    def action_open_pr(self, url: str):
        if re.fullmatch(r"https://github\.com/[A-Za-z0-9-]+/[A-Za-z0-9._-]+/pull/[1-9][0-9]*", url):
            self.open_url(url)

    def action_close(self):
        self.exit()

    def action_move(self, direction):
        scroll = self.query_one("#report", VerticalScroll)
        action = {"up": scroll.scroll_up, "down": scroll.scroll_down,
                  "page_up": scroll.scroll_page_up, "page_down": scroll.scroll_page_down,
                  "home": scroll.scroll_home, "end": scroll.scroll_end}[direction]
        action(animate=False)
