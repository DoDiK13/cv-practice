import argparse
import os
import sys
import pathlib

def cli_parser():
    parser = argparse.ArgumentParser(description="Утилита для использования фильтров на изображении")
    parser.add_argument(
        "-i", "--input",
        type=pathlib.Path,
        required=True,
        help="Путь файла (изображения), требующее модификации"
    )
    parser.add_argument(
        "-o", "--output",
        type=pathlib.Path,
        required=True,
        help="Путь модифицированного файла (изображения)"
    )
    parser.add_argument(
        "-f", "--filter",
        type=str,
        required=True,
        choices=["resize", "grayscale", "antique", "fadecolor", "tape", "matte", "noize", "neon"],
        help="Возможный тип фильтра"
    )
    parser.add_argument(
        "-p", "--params",
        nargs="+",
        default=[],
        help="Параметры для конкретного фильтра"
    )

    args = parser.parse_args()

    input_path : pathlib.Path = args.input
    if not input_path.exists() :
        parser.error(f"Файл не найден: {input_path}")
    if not input_path.is_file() :
        parser.error(f"Указанный путь не является путём к фалу: {input_path}")

    output_path : pathlib.Path = args.output
    if not output_path.parent.exists():
        parser.error(f"Папка для сохранения не существует: {output_path.parent}")

    filter_kwargs = {}
    for item in args.params:
        if not "=" in item:
            parser.error(f"Параметр должен быть в формате key=value, получено: {item!r}")

        key, value = item.split("=", 1)

        if value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        filter_kwargs[key] = value


    return input_path, output_path, args.filter, filter_kwargs
