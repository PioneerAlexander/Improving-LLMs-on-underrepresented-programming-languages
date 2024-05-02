"""
    Parse the kotlin files to body-signature pairs for the code completion task
"""

from typing import List


from src.parser.utils import SignatureBodyOutput, get_file_declarations, extract_function_declarations, \
    extract_body_and_signature_from_function_declaration, save_signature_body_pairs_to_json_file


def get_signature_body_pairs_from_kotlin_files(
        kt_filenames: str,
        not_parsed_filenames: str
) -> List[SignatureBodyOutput]:
    """
    Args:
        kt_filenames is the file with Kotlin filenames
    Returns:
        Collection of the signature-body pairs corresponded to the Kotlin functions
    """
    try:
        with open(kt_filenames, 'r', encoding='utf-8') as f:
            code_filenames = [name.strip() for name in f.readlines()]
    except FileNotFoundError:
        print(f'File with the name {kt_filenames} was not found in the root directory.')

    body_signature_pairs: List = []

    for name in code_filenames:
        try:
            declarations = get_file_declarations(filename=name)
            function_declarations_buffer = []

            function_declarations = extract_function_declarations(
                declarations,
                function_declarations_buffer
            )
        except Exception as g:
            # Dump names of all unsuccessfully parsed files to the specified file
            print(f"""Occurred an error {g} while trying to extract function declarations from
            the file {name}. Logging its name to the {not_parsed_filenames} buffer""")
            try:
                with open(not_parsed_filenames, "a", encoding='utf-8') as f:
                    f.write(name)
            except Exception as e:
                print(
                    f"""Occurred the error {e} while trying to log the unsuccessfully 
                    parsed files to {not_parsed_filenames} file""")
            continue

        body_signature_pairs.extend(
            [extract_body_and_signature_from_function_declaration(function_declaration)
             for function_declaration in function_declarations]
        )

    return body_signature_pairs


def main(
        kt_filenames: str,
        not_parsed_filenames: str,
        save_name: str,
):

    body_signature_pairs = get_signature_body_pairs_from_kotlin_files(
        kt_filenames,
        not_parsed_filenames,
    )

    save_signature_body_pairs_to_json_file(save_name, body_signature_pairs)


if __name__ == "__main__":
    main(
        "kt_filenames.txt",
        "not_parsed_filenames.txt",
        "body_signature_pairs.json"
    )
