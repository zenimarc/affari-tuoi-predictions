def amount_string_to_int(amount_str):
    """
    Convert an amount string to an integer.

    Parameters:
        amount_str (str): The amount string to convert.

    Returns:
        int: The integer value of the amount string.
    """
    amount_str = amount_str.replace('€', '').replace('.', '')
    return int(amount_str)


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def custom_similarity(s1: str, s2: str) -> float:
    """Compute a modified Levenshtein distance giving less penalty for the '€' symbol."""
    if s1 == s2:
        return 0.0

    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return float(len(s1))

    euro_penalty = 0.5
    digit_penalty = 1.0

    previous_row: list[float] = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row: list[float] = [i + 1]
        for j, c2 in enumerate(s2):
            if c1 == c2:
                cost = 0
            elif c1 == '€' and c2 != '€':
                cost = euro_penalty
            elif c1 != '€' and c2 == '€':
                cost = euro_penalty
            elif c1.isdigit() and c2.isdigit() and c1 != c2:
                cost = digit_penalty
            else:
                cost = 1

            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + cost

            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

