import re
import pdoc
import markdownify

import vimure as vm

from tqdm import tqdm

QMD_HEADER = (
    "---\n"
    "title: \"📚 Python package documentation\"\n"
    f"subtitle: \"VIMuRe {vm.VERSION} (latest)\"\n"
    "---\n"
)

vimure_model = 'src/python/vimure'

context = pdoc.Context()

main_module = pdoc.Module(vimure_model, context=context)
pdoc.link_inheritance(context)

def recursive_htmls(mod):
    yield mod.name, mod.html()
    for submod in mod.submodules():
        yield from recursive_htmls(submod)

if __name__ == '__main__':

    # Get HTML for all modules
    output_html = [{'module_name': output[0], 'module_html': output[1]}
                   for output in recursive_htmls(main_module)]

    # Convert HTML to Markdown
    output_md = [{'module_name': output['module_name'],
                  'module_md': markdownify.markdownify(output['module_html'], heading_style='ATX')}
                 for output in output_html 
                 if ('test' not in output['module_name'] and 
                     'extras' not in output['module_name'] and
                     'diagnostics' not in output['module_name'] and
                     output['module_name'] != 'vimure')]

    def remove_initial_css(module_md):
        # Regular expression pattern
        pattern = r"(?:(Module|Package) .*\n.*\n).*?(?=Module|Package|\Z)"

        # Extract the desired string
        result = re.search(pattern, module_md, re.DOTALL)

        return "# " + module_md[result.start():]

    # Remove the initial CSS from all output_md['module_md']
    for output in tqdm(output_md, desc="Removing initial CSS"):
        output['module_md'] = remove_initial_css(output['module_md'])

    def remove_index_section(module_md):
        # Find the index line using the re.search() function
        match = re.search(r".*# Index.*\n", module_md)

        return module_md[:match.start()]
    
    # Remove the index section from all output_md['module_md']
    for output in tqdm(output_md, desc="Removing index section"):
        output['module_md'] = remove_index_section(output['module_md'])

    def wrap_code_blocks(module_md):
        # Regular expression pattern
        code_regex = re.compile(r'(?P<header>Expand source code\n\n\n)```(?P<code>.*?)```', re.DOTALL)

        # Extract the desired string
        code_blocks = code_regex.findall(module_md)

        for header, code in code_blocks:
            wrapped_code = f"<details><summary>{header.strip()}</summary>\n```python\n{code}```\n</details>"
            module_md = module_md.replace(f'{header}```{code}```', wrapped_code)

        return module_md

    # Remove the initial CSS from all output_md['module_md']
    for output in tqdm(output_md, desc="Wrapping code blocks"):
        output['module_md'] = wrap_code_blocks(output['module_md'])

    def remove_details_tags(module_md):
        details_regex = re.compile(r'<details>.*?</details>', re.DOTALL)
        return details_regex.sub('', module_md)
    
    # Remove all <details> tags from all output_md['module_md']
    for output in tqdm(output_md, desc="Removing <details> tags"):
        output['module_md'] = remove_details_tags(output['module_md'])


    def fix_links(module_md):
        link_regex = re.compile(r'`\[([^]]+)\]\(([^)"]+)(?: "([^"]+)")?\)`')
        new_markdown_text = link_regex.sub(r'[`\1`](vimure.\2)', module_md)
        return new_markdown_text
    
    # Fix links in all output_md['module_md']
    for output in tqdm(output_md, desc="Fixing links"):
        output['module_md'] = fix_links(output['module_md'])

    def remove_extra_newlines(markdown_string):
        pattern = re.compile(r'\n{4,}')
        return re.sub(pattern, '\n\n', markdown_string)

    # Remove extra newlines from all output_md['module_md']
    for output in tqdm(output_md, desc="Removing extra newlines"):
        output['module_md'] = remove_extra_newlines(output['module_md'])

    def replace_escaped_characters(module_md):
        return re.sub(r'\\(\S)', r'\1', module_md)
    
    # Replace escaped characters in all output_md['module_md']
    for output in tqdm(output_md, desc="Replacing escaped characters"):
        output['module_md'] = replace_escaped_characters(output['module_md'])

    def format_functions_section(module_md):
        # Replace backticks surrounding function names with markdown headers
        module_md = re.sub(r'`def\s+(.*?)\((.*?)\)`', r'#### `\1`\n\n```python\ndef \1(\2):\n```', module_md)

        # Replace bolded "Parameters" and "Returns" with actual list items
        pattern = r"\*\*(.*?)\*\* : `(.*?)`\n(.*?)\n"
        replacement = r"- **`\g<1>`** : `\g<2>`\n\n    \g<3>\n\n"
        module_md = re.sub(pattern, replacement, module_md)
        module_md = re.sub(r'(\*\*Parameters\*\*|## Parameters)\n\n', r'**Parameters**\n\n', module_md)
        module_md = re.sub(r'(\*\*Returns\*\*|## Returns)\n\n', r'**Returns**\n\n', module_md)


        return module_md

    # Format functions section in all output_md['module_md']
    for output in tqdm(output_md, desc="Formatting functions section"):
        output['module_md'] = format_functions_section(output['module_md'])

    def format_classes(module_md):
        class_pattern = re.compile(r'`class\s+([\w\s]+)\n\((.*?)\)`')
        module_md = class_pattern.sub(r'### \g<1>\n\n```python\nclass \g<1>(\g<2>)\n```', module_md)

        module_md = re.sub(r'(\*\*Methods\*\*|### Methods)\n\n', r'#### Methods\n\n', module_md)
        module_md = re.sub(r'(\*\*Ancestors\*\*|### Ancestors)\n\n', r'#### Ancestors\n\n', module_md)
        module_md = re.sub(r'(\*\*Subclasses\*\*|### Subclasses)\n\n', r'#### Subclasses\n\n', module_md)
        module_md = re.sub(r'(\*\*Inherited members\*\*|### Inherited members)\n\n', r'#### Inherited members\n\n', module_md)

        return module_md
    
    # Format classes in all output_md['module_md']
    for output in tqdm(output_md, desc="Formatting classes"):
        output['module_md'] = format_classes(output['module_md'])

    def wrap_long_function_signatures(module_md):
        pattern = r'```python\s*\n(?:def)\s+(\w+)\((.*?)\):\s*'

        def format_params(match):
            func_name = match.group(1)
            params = match.group(2)

            embedded_links = re.findall(r'(\[(.*?)\]\(.*\))', params)
            if embedded_links is not None:
                for link in embedded_links:
                    params = params.replace(link[0], link[1])

            if (match.end() - match.start()) > 75:
                pattern = r',\s*(?![^()]*\))'
                params_str = re.sub(pattern, ',\n    ', params)
                return f'\n```python\ndef {func_name}(\n    {params_str}\n):\n'
            else:
                return f'\n```python\ndef {func_name}({params}):\n'
            
        return re.sub(pattern, format_params, module_md)

    # Wrap long signatures in all output_md['module_md']
    for output in tqdm(output_md, desc="Wrapping long signatures"):
        output['module_md'] = wrap_long_function_signatures(output['module_md'])


    def wrap_long_class_signatures(module_md):
        pattern = r'```python\s*\nclass\s+(\w+)\((.*?)\)\s*'

        def format_params(match):
            class_name = match.group(1)
            params = match.group(2)
            if (match.end() - match.start()) > 75:
                pattern = r',\s*(?![^()]*\))'
                params_str = re.sub(pattern, ',\n    ', params)
                return f'\n```python\nclass {class_name}(\n    {params_str}\n):\n'
            else:
                return f'\n```python\nclass {class_name}({params}):\n'

        return re.sub(pattern, format_params, module_md)

    # Wrap long class signatures in all output_md['module_md']
    for output in tqdm(output_md, desc="Wrapping long class signatures"):
        output['module_md'] = wrap_long_class_signatures(output['module_md'])

    # Wrap each one of output_md['module_md'] in a <details> tag
    for output in tqdm(output_md, desc="Wrapping in <details> tags"):
        output['module_md'] = f"<details><summary>Module `{output['module_name']}`</summary>\n\n{output['module_md']}\n\n</details>\n\n"
    
    # Concatenate all output_md['module_md'] into a single string
    output_md = ''.join([output['module_md'] for output in output_md])

    # Add QMD_HEADER to the top of all output_md['module_md']
    output_md = QMD_HEADER + output_md

    # Write each module to a separate file
    with open(f"docs/latest/pkg-docs/python.qmd", 'w') as f:
        f.write(output_md)