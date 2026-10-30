import asyncio

from specguard.cli.picker import Picker


def test_arrow_keys_select_item_and_escape_cancels():
    async def exercise():
        picker = Picker("Выбор", [("Первый", "a"), ("Второй", "b"), ("Назад", None)])
        async with picker.run_test() as pilot:
            await pilot.press("down", "enter")
        assert picker.return_value == 1

        picker = Picker("Выбор", [("Первый", "a")])
        async with picker.run_test() as pilot:
            await pilot.press("escape")
        assert picker.return_value is None

    asyncio.run(exercise())


def test_f_filters_directory_and_selects_original_value():
    from textual.widgets import Input, OptionList
    async def exercise():
        picker = Picker('Folder', [('alpha.md', 'a'), ('beta.md', 'b')], searchable=True)
        async with picker.run_test() as pilot:
            await pilot.press('f', 'b', 'e', 't', 'a')
            await pilot.pause()
            assert picker.query_one(Input).value == 'beta'
            assert picker.query_one(OptionList).option_count == 1
            await pilot.press('enter', 'enter')
        assert picker.return_value == 1
    asyncio.run(exercise())


def test_d_removes_saved_item_and_other_items_remain_selectable():
    async def exercise():
        deleted = []
        picker = Picker('Repos', [('Saved', 'owner/saved'), ('Other', 'owner/other')],
                        deletable=['owner/saved'], on_delete=deleted.append)
        async with picker.run_test() as pilot:
            await pilot.press('d')
            await pilot.pause()
            assert deleted == ['owner/saved']
            await pilot.press('d', 'enter')
        assert deleted == ['owner/saved']
        assert picker.return_value == 1
    asyncio.run(exercise())
