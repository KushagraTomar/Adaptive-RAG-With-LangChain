import re

def convert_section_headers(text):
    lines = text.splitlines()
    output = []

    for line in lines:
        match = re.match(
            r'^\*\*(\d+(?:\.\d+)*)\*\*\s+\*\*(.*?)\*\*\s*$',
            line.strip()
        )
        if match:
            number = match.group(1)
            title = match.group(2)
            level = number.count(".") + 1
            prefix = "#" * level
            output.append(f"{prefix} {number} {title}")
        else:
            output.append(line)

    return "\n".join(output)