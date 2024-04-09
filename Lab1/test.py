with open("TCGAlabels", "r") as file:
    lines = file.readlines()

# Initialize a dictionary to store frequencies
frequency = {}

# Iterate through each line in the file
for line in lines:
    # Extract the category from each line
    category = line.strip().split()[1]

    # Update the frequency count for the category
    if category in frequency:
        frequency[category] += 1
    else:
        frequency[category] = 1

# Print the frequencies
for category, count in frequency.items():
    print(f"Category {category}: {count} occurrences")
