import os


def get_short_path(full_path: str) -> str:
    """Converts a full absolute path to 'project_folder/relative_path'."""
    abs_path = os.path.abspath(full_path)
    # Get relative path starting from parent directory of project root
    return os.path.relpath(abs_path, start=os.path.dirname(abs_path))
