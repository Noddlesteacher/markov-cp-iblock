"""
Small sanity checks for the Markov CP helper routines.

Run from the repository root with:

    python sanity_checks.py
"""

from markov_cp_routines import generate_i_block_permutations


def check_i_block_permutations():
    """Check that exact i-block permutations do not duplicate the identity."""
    sequence = [1, 2, 3, 1, 3]
    i = 3

    permuted_sequences = generate_i_block_permutations(
        sequence,
        i,
        max_permutations=None,
    )

    sequence_keys = [tuple(seq) for seq in permuted_sequences]
    identity_key = tuple(sequence)

    number_unique = len(set(sequence_keys))
    identity_count = sequence_keys.count(identity_key)

    print("i-block permutation sanity check")
    print("--------------------------------")
    print(f"sequence: {sequence}")
    print(f"i: {i}")
    print(f"permuted sequences: {permuted_sequences}")
    print(f"number returned: {len(permuted_sequences)}")
    print(f"number unique: {number_unique}")
    print(f"identity count: {identity_count}")

    assert number_unique == len(permuted_sequences)
    assert identity_count == 1


def main():
    check_i_block_permutations()
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
