import os
from collections import defaultdict
from fuzzywuzzy import fuzz

def parse_filename(filename):
    """Parse filename into artist, title, dance."""
    base = os.path.splitext(filename)[0].strip()
    parts = [p.strip() for p in base.split('-')]
    if len(parts) < 2:
        return None, None, None  # Not enough parts

    dance = parts[-1]
    artist = parts[0]
    title = '-'.join(parts[1:-1])

    return artist.lower(), title.lower(), dance.lower()

def find_duplicates(directory, similarity_threshold=80):
    """Find exact and potential duplicate files."""
    exact_duplicates = defaultdict(list)
    ungrouped = []

    # Step 1: Parse and group exact matches
    for filename in os.listdir(directory):
        if not filename.endswith(('.mp3', '.wav')):
            continue

        ungrouped.append(filename)

    # Step 2: Group exact duplicates
    # duplicate_groups = {k: v for k, v in exact_duplicates.items() if len(v) > 1}

    # Step 3: Fuzzy match ungrouped files
    potential_duplicates = defaultdict(list)
    for i, f1 in enumerate(ungrouped):
        for j, f2 in enumerate(ungrouped[i+1:], i+1):
            a1, t1, d1 = parse_filename(f1)
            a2, t2, d2 = parse_filename(f2)

            # Skip if parsing failed
            if not all([a1, t1, d1, a2, t2, d2]):
                continue

            # Fuzzy match artist + title + dance
            score = (
                fuzz.token_sort_ratio(f"{a1} - {t1} - {d1}", f"{a2} - {t2} - {d2}")
            )

            if score >= similarity_threshold:
                key = (f1, f2)
                title_score = fuzz.token_sort_ratio(f"{t1}", f"{t2}")
                if title_score >= similarity_threshold:
                    potential_duplicates[key].append(title_score)
                else:
                    exact_duplicates[key].append(score)

    return exact_duplicates, potential_duplicates

# --- Main Execution ---
if __name__ == "__main__":
    directory = "sources"  # Replace with your directory
    exact_duplicates, potential_duplicates = find_duplicates(directory)

    print("✅ Exact Duplicates:")
    for (f1, f2), scores in exact_duplicates.items():
        avg_score = sum(scores) // len(scores)
        print(f"\nSimilarity: {avg_score}%")
        print(f" - {f1}")
        print(f" - {f2}")

    print("\n⚠️ Potential Duplicates (fuzzy match):")
    for (f1, f2), scores in potential_duplicates.items():
        avg_score = sum(scores) // len(scores)
        print(f"\nSimilarity: {avg_score}%")
        print(f" - {f1}")
        print(f" - {f2}")