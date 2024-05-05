"""
    Parse the kotlin files to body-signature pairs for the code completion task
"""
from src.parser.utils import save_signature_body_pairs_to_json_file, get_signature_body_pairs_from_kotlin_files


def main(
        kt_filenames: str,
        not_parsed_filenames: str,
        save_name: str,
        /,  # Mark the end of positional arguments
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
