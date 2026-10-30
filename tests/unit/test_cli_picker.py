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
