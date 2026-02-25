from typing import Optional
import unicodedata
import sys
import traceback


def find_approx_substring(needle: str, haystack: str) -> Optional[str]:
    """Find `needle` inside `haystack` ignoring case, punctuation and whitespace.

    If found, returns the substring of `haystack` (with original case,
    punctuation and whitespace) that corresponds to the match. If not found
    returns `None`.

    Algorithm:
    - Normalize both strings by keeping only alphanumeric characters and
      lowercasing them.
    - Build a mapping from normalized haystack positions back to original
      haystack indices so we can return the original slice.
    """
    def _decompose_strip(ch: str) -> str:
        d = unicodedata.normalize("NFKD", ch)
        return "".join(c for c in d if unicodedata.category(c) != "Mn")

    # Normalize needle (decompose accents, strip combining marks)
    needle_norm = "".join(
        c.lower()
        for ch in needle
        for c in _decompose_strip(ch)
        if c.isalnum()
    )
    if not needle_norm:
        return None

    # Build normalized haystack and mapping to original indices
    hay_norm_chars = []
    map_to_orig = []  # map normalized index -> original index in haystack
    for idx, ch in enumerate(haystack):
        for c in _decompose_strip(ch):
            if c.isalnum():
                hay_norm_chars.append(c.lower())
                map_to_orig.append(idx)

    hay_norm = "".join(hay_norm_chars)
    pos = hay_norm.find(needle_norm)
    if pos == -1:
        return None

    start_orig = map_to_orig[pos]
    end_orig = map_to_orig[pos + len(needle_norm) - 1]
    return haystack[start_orig : end_orig + 1]


if __name__ == "__main__":
    examples = [
        # basic punctuation/case
        ("Hello, World!", "helloworld", "Hello, World"),
        ("The quick, brown fox.", "quickbrownfox", "quick, brown fox"),
        ("abc-def", "abcdef", "abc-def"),
        # no match
        ("No match here", "nomatchx", None),
        # surrounding whitespace in haystack: function currently trims leading/trailing
        (" spaced   out ", "spacedout", "spaced   out"),
        # unicode accents
        ("Café-au-lait", "cafeaulait", "Café-au-lait"),
        # needle empty -> normalized empty -> expect None
        ("anything", "   ", None),
        # needle longer than haystack -> None
        ("short", "shortandlong", None),
        # match at end
        ("ends with match!", "withmatch", "with match"),
        # match with internal punctuation differences
        ("a(b)c.d,e", "abcde", "a(b)c.d,e"),
        # repeated / overlapping: should find first occurrence
        ("xx-xx-xx", "xxxx", "xx-xx"),
        # mixed unicode + punctuation
        ("naïve-example_test", "naiveexampletest", "naïve-example_test"),
    ]

    failures = 0
    for i, (hay, needle, expected) in enumerate(examples, start=1):
        try:
            found = find_approx_substring(needle, hay)
            print(f"[{i}] hay: {hay!r} | needle: {needle!r} -> found: {found!r}")
            if expected is None:
                assert found is None, f"expected None but got {found!r}"
            else:
                assert found == expected, f"expected {expected!r} but got {found!r}"
        except Exception:
            failures += 1
            print(f"[{i}] TEST FAILURE for needle={needle!r} in hay={hay!r}")
            traceback.print_exc()
            print("--- continuing to next test ---\n")

    if failures:
        print(f"{failures} test(s) failed.")
        sys.exit(1)

    print("All tests passed.")
