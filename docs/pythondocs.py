import re
import pdoc
import markdownify

from tqdm import tqdm

QMD_HEADER = (
    "---\n"
    "title: \"📚 Python package documentation\"\n"
    "subtitle: \"VIMuRe v0.1 (latest)\"\n"
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
                 for output in output_html]

    def remove_initial_css(module_md):
        # Regular expression pattern
        pattern = r"(?:(Module|Package) .*\n.*\n).*?(?=Module|Package|\Z)"

        # Extract the desired string
        result = re.search(pattern, module_md, re.DOTALL)

        return module_md[result.start():]

    # Remove the initial CSS from all output_md['module_md']
    for output in tqdm(output_md, desc="Removing initial CSS"):
        output['module_md'] = remove_initial_css(output['module_md'])

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

    # Add QMD_HEADER to the top of all output_md['module_md']
    for output in tqdm(output_md, desc="Adding QMD header"):
        output['module_md'] = QMD_HEADER + output['module_md']

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

    # Write each module to a separate file
    for output in tqdm(output_md, desc="Writing to file"):
        with open(f"docs/latest/pdocs/{output['module_name']}.qmd", 'w') as f:
            f.write(output['module_md'])