import re
from html.parser import HTMLParser


class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()


def format_text_to_html(text):

    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'^\*(.*)$', r'⚪ \1', text, flags=re.MULTILINE)

    text = re.sub(r'^#{2,}(.*)$', lambda m: '{}'.format(m.group(1).strip().upper()), text, flags=re.MULTILINE)
    text = re.sub(r'^#(.*)$', lambda m: '<b>{}</b>'.format(m.group(1).strip().upper()), text, flags=re.MULTILINE)

    language_tags = {
        'python': '<pre language="python">',
        'javascript': '<pre language="javascript">',
        'java': '<pre language="java">',
        # Add more languages as needed
    }
    end_tag = '</pre>'

    # Extract the language from the code block delimiter
    while '```' in text:
        # Extract the language from the code block delimiter
        match = re.search(r'```(\w+)', text)
        if match:
            language = match.group(1)
            if language in language_tags:
                start_tag = language_tags[language]
                text = text.replace("```" + language, start_tag, 1)
                text = text.replace("```", end_tag, 1)
            else:
                text = text.replace("```" + language, '<code>', 1)
                text = text.replace("```", '</code>', 1)
        else:
            text = text.replace("```", '<code>', 1)
            text = text.replace("```", '</code>', 1)

    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)

    parser = MyHTMLParser()
    parser.feed(text)

    if parser.stack:
        for tag in parser.stack:
            text = re.sub(r'<{}[^>]*>'.format(tag), '', text)

    text = re.sub(r'[^\x00-\x7F]+', '', text)

    return text
