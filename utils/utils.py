import locale
def format_number(num, use_unit=True):
    locale.setlocale(locale.LC_ALL, '')
    formatted_num = locale.format_string('%d', num, grouping=True)
    if use_unit:
        if num >= 1000000000.0:
            return f'{num / 1000000000.0:.2f}B ({formatted_num})'
        elif num >= 1000000.0:
            return f'{num / 1000000.0:.2f}M ({formatted_num})'
        elif num >= 1000.0:
            return f'{num / 1000.0:.2f}K ({formatted_num})'
    return formatted_num
