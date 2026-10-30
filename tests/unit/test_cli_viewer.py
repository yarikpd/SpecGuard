import asyncio
from rich.text import Text
from specguard.cli.viewer import ReportViewer
from textual.containers import VerticalScroll


def test_long_report_scrolls_both_directions_and_closes():
    async def exercise():
        viewer = ReportViewer(Text('\n'.join(f'Line {i}' for i in range(200))))
        async with viewer.run_test(size=(80, 24)) as pilot:
            scroll = viewer.query_one('#report', VerticalScroll)
            assert scroll.scroll_y == 0
            await pilot.press('pagedown')
            await pilot.pause()
            assert scroll.scroll_y > 0
            await pilot.press('end')
            await pilot.pause()
            assert scroll.scroll_y == scroll.max_scroll_y
            await pilot.press('home')
            await pilot.pause()
            assert scroll.scroll_y == 0
            await pilot.press('escape')
        assert not viewer.is_running
    asyncio.run(exercise())
