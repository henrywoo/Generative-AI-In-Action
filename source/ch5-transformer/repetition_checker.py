import re

def check_repetitive_sequences(text):
    # Set a threshold for minimum sequence length to avoid tiny sequences
    min_seq_len = 1

    # Iterate over possible sequence lengths
    for seq_len in range(min_seq_len, len(text) // 50):
        # Use regex to find sequences of the current length repeated more than 50 times
        pattern = r'(.{' + str(seq_len) + r'})\1{49,}'
        match = re.search(pattern, text)
        
        if match:
            return True, match.group(0)

    return False, None

# Example usage
text = "This is a test text with 1" + "47" * 80
has_repetitive_sequences, repetitive_sequence = check_repetitive_sequences(text)

if has_repetitive_sequences:
    print(f"Found a repetitive sequence: {repetitive_sequence}")
else:
    print("No repetitive sequence found.")
