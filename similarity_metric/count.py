import re

def count_function_lines(file_path):
    with open(file_path, 'r') as file:
        code = file.read()

    function_pattern = re.compile(r'def (\w+)\(I\):\s*(.*?)\s*return O', re.DOTALL)
    function_line_counts = {}

    for match in function_pattern.finditer(code):
        function_name = match.group(1)
        function_name = function_name.split('_', 1)[-1] + '.json'
        function_body = match.group(2)

        lines = function_body.strip().splitlines()

        function_line_counts[function_name] = len(lines)
    
    return function_line_counts
