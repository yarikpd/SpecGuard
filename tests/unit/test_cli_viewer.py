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


def test_pr_title_link_survives_history_and_opens_on_click():
    from specguard.cli.render import render_report
    from specguard.models.schema import AuditReport, GitContext
    report = AuditReport(audit_id='audit_pr', target_repo='owner/repo', target_ref='7',
                         saved_git_context=GitContext(target_repo='owner/repo', target_ref='PR#7',
                                                      head_sha='abc', pr_number=7, pr_title='Add validation [bold]'))
    report = AuditReport.model_validate_json(report.model_dump_json())
    header = render_report(report).renderables[0].renderable
    assert 'PR #7 · Add validation [bold]' in header.plain
    assert any(span.style.link == 'https://github.com/owner/repo/pull/7' for span in header.spans)
    async def exercise():
        viewer = ReportViewer(header)
        opened = []
        viewer.open_url = lambda url: opened.append(url)
        async with viewer.run_test(size=(100, 24)) as pilot:
            await pilot.click('#content', offset=(3, 0))
            await pilot.pause()
            assert opened == ['https://github.com/owner/repo/pull/7']
    asyncio.run(exercise())
