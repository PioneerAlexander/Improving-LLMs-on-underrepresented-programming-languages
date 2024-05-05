"""
    Preprocessing dataset functions
"""

import re
import pandas as pd


def read_dataframe_from_json(filename: str, lines: bool = False) -> pd.DataFrame:
    """
    Returns DataFrame with columns for the signature and bodies of the functions
    """

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            df = pd.read_json(f, lines=lines)
            return df

    except IOError as e:
        raise IOError(f"The exception '{type(e)}: {e}' occurred while trying to read json dataset "
                      f"of signatures and bodies of the functions. ") from e


def discard_functions_with_long_bodies(max_len: int, df: pd.DataFrame) -> pd.DataFrame:
    """
        Returns a new version of the DataFrame where the body is not longer than defined max_len
    """
    return df[df["body"].str.len() <= max_len]


def get_indent(s: str) -> int:
    """
       Returns the indentation for the input line
    """
    match = re.match(r' *', s)
    return len(match.group(0))


def replace_indentation_and_eol_symbols(
        body: str,
        indent_token: str = "<INDENT>",
        dedent_token: str = "<DEDENT>",
        end_of_line_token: str = "<EOL>",
):
    """
        This function applied to body returns the new version of it where all indentations
        and end of line symbols are replaced with special tokens.
    """
    result: str = ""
    indent_stack = [0]  # Initialize indentation stack
    for line in body.splitlines():
        indent = get_indent(line)

        if indent > indent_stack[-1]:
            indent_stack.append(indent)  # If we made a new indent, add new last indent to the stack
            result += indent_token + line[indent:] + end_of_line_token
        elif indent < indent_stack[-1]:
            dedent_string = ""

            while indent < indent_stack[-1]:
                dedent_string += dedent_token
                indent_stack.pop()  # Pop all indent values for each dedent

            result += dedent_string + line[indent:] + end_of_line_token
        else:  # Do not add token if indentation has not changed
            result += line[indent:] + end_of_line_token

    return result


def decode_function_body(
        body: str,
        indent_token: str = "<INDENT>",
        dedent_token: str = "<DEDENT>",
        end_of_line_token: str = "<EOL>",
):
    """
    This function is an inverse transformation which replaces all special tokens in decoded body to the spaces and new
    line symbol respectively
    """

    body_lines = body.split(end_of_line_token)

    res_string = ""
    indent_cnt = 0
    for idx, line in enumerate(body_lines[:-1]):
        if line.startswith(indent_token):
            indent_cnt += 4
            res_string += line.replace(indent_token, " " * indent_cnt)

        elif line.startswith(dedent_token):
            while line.startswith(dedent_token):
                line = line.removeprefix(dedent_token)
                indent_cnt -= 4
            res_string += " " * indent_cnt + line

        else:
            res_string += " " * indent_cnt + line
        if idx != len(body_lines[:-1]) - 1:
            res_string += "\n"

    return res_string
