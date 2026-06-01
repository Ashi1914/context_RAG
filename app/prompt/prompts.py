"""
Prompts for content generation.
"""

def get_prompt(content_type, tone, topic, keywords):
    """
    Get a prompt for content generation.
    """
    return f"Write a {content_type} in a {tone} tone about {topic}. Include the following keywords: {keywords}."
