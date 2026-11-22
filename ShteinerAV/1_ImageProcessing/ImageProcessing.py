from cli_argument_parser import cli_parser
from filters import ImageFilter
from image_io import read_image, write_image
from filters import ImageFilter
import sys

def main() :
    try:
        input_path, output_path, filter, filter_kwargs = cli_parser()
        image = read_image(input_path)
    except Exception as e:
        print(f"Ошибка данных: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        usFilter : ImageFilter = ImageFilter.get_filter(filter, **filter_kwargs)
        res_image = usFilter.apply_filter(image)
    except Exception as e:
        print(f"Ошибка данных: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        write_image(output_path, res_image)
    except Exception as e:
        print(f"Ошибка данных: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__" :
    main()