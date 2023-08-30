import json
import os
import re

def process_file_name(file_name):
    # Convert file_name to lowercase and replace special characters with underscores
    file_name = re.sub(r'\W+', '_', file_name).lower()
    # Replace multiple underscores with a single underscore
    file_name = re.sub(r'_+', '_', file_name)
    # Strip underscores at the beginning and end
    file_name = file_name.strip('_')
    # Change "_py" back to ".py" only if it's at the end of the filename
    if file_name.endswith('_py'):
        file_name = file_name[:-3] + '.py'

    return file_name

def process_extracted_name(title, index):
    return f"{index}_{title.group(1).strip().lstrip('#0123456789. ').replace(' ', '_')}.py"

def waterfall_title(cell_content, i):
    cell_name = re.search(r'^cell_name ? = ?[\'"](.*)[\'"]', cell_content)
    if cell_name:
        file_name = process_extracted_name(cell_name, i)
        return file_name
    title = re.search(r'#@title\s+(.*)[,\n]', cell_content)
    if title:
        file_name = process_extracted_name(title, i)
        return file_name
    markdown_title = re.search(r'^#@markdown\s+(.*)[,\n]', cell_content, re.MULTILINE)
    if markdown_title:
        file_name = process_extracted_name(markdown_title, i)
        return file_name
    comment_title = re.search(r'^#\s*(.*)[,\n]', cell_content, re.MULTILINE)
    if comment_title:
        file_name = process_extracted_name(comment_title, i)
        return file_name
    return f'{i}_unnamed_cell.py'


def split_notebook_cells(notebook_path, output_dir):
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    counter = 1
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            cell_content = ''.join(cell['source']).strip()
            if not cell_content:
                continue

            file_name = waterfall_title(cell_content, counter)
            file_name = process_file_name(file_name)

            with open(os.path.join(output_dir, file_name), 'w') as cell_file:
                cell_file.write(cell_content)
            counter += 1

if __name__ == '__main__':
    notebook_path = 'stable_warpfusion.ipynb'  # replace with your notebook file path
    output_dir = 'gen_src'  # replace with the desired output directory
    split_notebook_cells(notebook_path, output_dir)
