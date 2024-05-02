"""
    This module contains utility functions for parsing the dataset
"""
from dataclasses import dataclass
from typing import List, Sequence, Optional

import pandas as pd
from kopyt import Parser
from kopyt.node import FunctionDeclaration, Declaration
import timeout_decorator


@dataclass
class SignatureBodyOutput:
    """
        Dataclass for the output definition
    """
    signature: str
    body: str


def extract_function_declarations(
        declarations: Sequence[Declaration],
        result: Optional[List[FunctionDeclaration]] = None
) -> List[FunctionDeclaration]:
    """
        Recursive function which extracts nested function declarations \
        from parsed kotlin file
    """
    if result is None:
        result = []

    for declaration in declarations:
        if isinstance(declaration, FunctionDeclaration):
            result.append(declaration)

        if hasattr(declaration, "body") and hasattr(declaration.body, "members"):
            extract_function_declarations(declaration.body.members, result)

    return result


@timeout_decorator.timeout(30, timeout_exception=TimeoutError)
def get_file_declarations(filename: str) -> Sequence[Declaration]:
    """
        Get declarations from the .kt file
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            code = f.read()
    except FileNotFoundError:
        print(f'File with the name {filename} was not found in the root directory.')

    return Parser(code).parse_kotlin_file().declarations


def extract_body_and_signature_from_function_declaration(
        function_declaration: FunctionDeclaration
) -> SignatureBodyOutput:
    """
        Returns the specified data class output for body and signature of the function
    """

    body = str(function_declaration.body)
    signature = str(function_declaration).removesuffix(body)

    return SignatureBodyOutput(signature, body)


def from_signature_body_pairs_to_dataframe(
        signature_body_pairs: List[SignatureBodyOutput]
) -> pd.DataFrame:
    """
        Creates pandas DataFrame from signature-body pairs collected before
    """
    return pd.DataFrame([signature_body_pair.__dict__ for signature_body_pair in signature_body_pairs])


def save_signature_body_pairs_to_json_file(filename: str, signature_body_pairs: List[SignatureBodyOutput]) -> None:
    """
     Saves collected signature body pairs to the json with the defined filename.
    """

    df = from_signature_body_pairs_to_dataframe(signature_body_pairs)

    df.to_json(path_or_buf=filename)
