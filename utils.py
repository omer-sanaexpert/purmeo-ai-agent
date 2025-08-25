import re
from html import unescape

def strip_html(content: str) -> str:
    """Removes code blocks, HTML tags, and unnecessary whitespace from the given content."""
    
    # Remove code blocks (content between triple backticks)
    content = re.sub(r'```[\s\S]*?```', '', content)
    
    # Remove style attributes
    content = re.sub(r'style="[^"]*"', '', content)
    
    # Remove HTML comments
    content = re.sub(r'<!--[\s\S]*?-->', '', content)
    
    # Remove script tags and their content
    content = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', content, flags=re.IGNORECASE)
    
    # Remove style tags and their content
    content = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', content, flags=re.IGNORECASE)
    
    # Replace common block elements with newlines
    content = re.sub(r'</(div|p|h[1-6]|table|tr|li)>', '\n', content, flags=re.IGNORECASE)
    
    # Replace <br> tags with newlines
    content = re.sub(r'<br[^>]*>', '\n', content, flags=re.IGNORECASE)
    
    # Remove all remaining HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    
    # Decode HTML entities
    content = unescape(content)  # Converts &nbsp;, &amp;, &lt;, &gt;, etc.

    # Clean up whitespace
    content = re.sub(r'\n\s*\n', '\n', content)  # Remove multiple empty lines
    content = content.strip()  # Trim start and end
    content = re.sub(r'[ \t]+', ' ', content)  # Normalize spaces and tabs
    
    # Normalize lines
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    return '\n'.join(lines)

def count_tokens(text: str) -> int:
    """Count the number of tokens in a given text."""
    # Count tokens before creating a message
    print(text)
    count = client.beta.messages.count_tokens(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": text}],
    )
    return count
