import pandas as pd

from src.dataset.preprocess import discard_functions_with_long_bodies, replace_indentation_and_eol_symbols, \
    decode_function_body

from src.parser.utils import from_signature_body_pairs_to_dataframe, SignatureBodyOutput


def test_body_pairs_to_dataframe():
    sig_body_pair1 = SignatureBodyOutput("abc", "cdr")

    assert isinstance(from_signature_body_pairs_to_dataframe([sig_body_pair1]), pd.DataFrame)

    print(from_signature_body_pairs_to_dataframe([sig_body_pair1]))


def test_discard_functions_with_long_bodies():
    sig_body_pair1 = SignatureBodyOutput("abc", "cdr")
    sig_body_pair3 = SignatureBodyOutput("abc", "cdrjdgjlkjlgjklajfrjrklgjkgjelkrgjelrg")
    sig_body_pair2 = SignatureBodyOutput("abc", "kglglggfkllkgk")

    df = from_signature_body_pairs_to_dataframe([sig_body_pair1, sig_body_pair2, sig_body_pair3])

    assert from_signature_body_pairs_to_dataframe(
        [sig_body_pair1, sig_body_pair2]).equals(discard_functions_with_long_bodies(20, df))

    assert from_signature_body_pairs_to_dataframe(
        [sig_body_pair1]).equals(discard_functions_with_long_bodies(5, df))

    assert from_signature_body_pairs_to_dataframe(
        [sig_body_pair1, sig_body_pair2, sig_body_pair3]).equals(discard_functions_with_long_bodies(100, df))


def test_replace_indentation_and_eol_symbols():
    body1 = """{
    val original = super.visitMethod(access, name, desc, signature, exceptions)
    return if (predicate(name, desc)) {
        assert(!visited)
        visited = true
        transform(original)
    }
    else {
        original
    }
}"""

    answer1 = ("{<EOL><INDENT>val original = super.visitMethod(access, name, desc, signature, exceptions)<EOL>"
               "return if (predicate(name, desc)) {<EOL><INDENT>assert(!visited)<EOL>visited = true<EOL>transform"
               "(original)<EOL><DEDENT>}<EOL>else {<EOL><INDENT>original<EOL><DEDENT>}<EOL><DEDENT>}<EOL>")
    assert replace_indentation_and_eol_symbols(body1) == answer1


def test_decode_function_body():

    body1 = """{
    val original = super.visitMethod(access, name, desc, signature, exceptions)
    return if (predicate(name, desc)) {
        assert(!visited)
        visited = true
        transform(original)
    }
    else {
        original
    }
}"""

    answer1 = ("{<EOL><INDENT>val original = super.visitMethod(access, name, desc, signature, exceptions)<EOL>"
               "return if (predicate(name, desc)) {<EOL><INDENT>assert(!visited)<EOL>visited = true<EOL>transform"
               "(original)<EOL><DEDENT>}<EOL>else {<EOL><INDENT>original<EOL><DEDENT>}<EOL><DEDENT>}<EOL>")
    assert decode_function_body(answer1) == body1
    assert decode_function_body(replace_indentation_and_eol_symbols(body1)) == body1
