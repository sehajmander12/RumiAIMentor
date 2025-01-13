import pdfplumber
import json
import matplotlib.pyplot as plt

chapter_map = {
    "ON THE TAVERN": "The Tavern: Whoever Brought Me Here Will Have to Take Me Home",
    "ON BEWILDERMENT": "Bewilderment: I Have Five Things to Say",
    "ON SILENCE": "Emptiness and Silence: The Night Air",
    "ON SPRING GIDDINESS": "Spring Giddiness: Stand in the Wake of This Chattering and Grow Airy",
    "ON SEPARATION": "Feeling Separation: Don’t Come Near Me",
    "ON THE DESIRE-BODY": "Controlling the Desire-Body: How Did You Kill Your Rooster, Husam?",
    "ON SOHBET": "Sohbet: Meetings on the Riverbank",
    "ON BEING A LOVER": "Being a Lover: The Sunrise Ruby",
    "ON THE PICKAXE": "The Pickaxe: Getting to the Treasure Beneath the Foundation",
    "ON FLIRTATION": "Art as Flirtation with Surrender: Wanting New Silk Harp Strings",
    "ON UNION": "Union: Gnats Inside the Wind",
    "ON THE SHEIKH": "The Sheikh: I Have Such a Teacher",
    "ON ELEGANCE": "Recognizing Elegance: Your Reasonable Father",
    "ON HOWLING": "The Howling Necessity: Cry Out in Your Weakness",
    "ON THE UNSEEN": "Teaching Stories: How the Unseen World Works",
    "ON ROUGHNESS": "Rough Metaphors: More Teaching Stories",
    "ON SOLOMON": "Solomon Poems: The Far Mosque",
    "ON GAMBLING": "The Three Fish: Gamble Everything for Love",
    "ON JESUS": "Jesus Poems: The Population of the World",
    "ON BAGHDAD": "In Baghdad, Dreaming of Cairo: More Teaching Stories",
    "ON THE FRAME": "Beginning and End: The Stories That Frame the Mathnawi",
    "ON CHILDREN RUNNING THROUGH": "Green Ears Everywhere: Children Running Through",
    "ON BEING WOVEN": "Being Woven: Communal Practice",
    "ON SECRECY": "Wished-For Song: Secret Practices",
    "ON MAJESTY": "Majesty: This We Have Now",
    "ON EVOLVING": "Evolutionary Intelligence: Say I Am You",
    "ON THE TURN": "The Turn: Dance in Your Blood"
}


def pdf_to_txt(pdf_path, txt_path):
    """
    Convert a PDF to a plain text file.
    
    Args:
        pdf_path (str): Path to the input PDF.
        txt_path (str): Path to save the output TXT file.
    """
    with pdfplumber.open(pdf_path) as pdf:
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            for page in pdf.pages:
                text = page.extract_text()  # Extract text from the page
                if text:
                    txt_file.write(text + "\n\n")  # Write text to file with a blank line between pages

    print(f"Text has been successfully extracted to '{txt_path}'.")

def remove_page_numbers(input_file, output_file):
    """
    Removes lines that contain only numbers from a text file.
    
    Args:
        input_file (str): Path to the input TXT file.
        output_file (str): Path to save the cleaned TXT file.
    """
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            # Strip whitespace from the line
            stripped_line = line.strip()
            
            # Check if the line is numeric and skip it if true
            if not stripped_line.isdigit():
                outfile.write(line)  # Write the line if it's not a page number

    print(f"Page numbers removed. Cleaned text saved to '{output_file}'.")

def process_blank_lines(input_file, output_file):
    """
    First removes all blank lines, then adds a blank line before lines with all capital letters.
    
    Args:
        input_file (str): Path to the input TXT file.
        output_file (str): Path to save the processed TXT file.
    """
    # Step 1: Remove all blank lines
    lines_without_blanks = []
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():  # Skip blank lines
                lines_without_blanks.append(line.strip())  # Append non-blank lines

    # Step 2: Add blank lines before lines with all capital letters
    processed_lines = []
    for i, line in enumerate(lines_without_blanks):
        if line.isupper():  # Check if the line is in all caps
            processed_lines.append("")  # Add a blank line
        processed_lines.append(line)  # Add the current line

    # Write the processed lines to the output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for line in processed_lines:
            outfile.write(line + "\n")

    print(f"Processing complete. Result saved to '{output_file}'.")

def is_chapter_line(line: str) -> bool:
    """
    Returns True if this stripped line is a known chapter heading 
    like 'ON THE TAVERN', 'ON BEWILDERMENT', etc.
    """
    return line.strip() in chapter_map

def is_poem_title(current_line: str, prev_line: str, chapter_map: dict) -> bool:
    """
    Returns True if:
    1) The previous line is empty (blank),
    2) The current line is in all caps (after removing punctuation/spaces),
    3) The current line is not a known chapter heading in chapter_map.
    """
    # 1) Check if previous line is blank
    if prev_line.strip():
        return False

    current_line_stripped = current_line.strip()
    if not current_line_stripped:
        return False  # current line is also empty, can't be a title

    # 2) Check if current line is in the chapter_map (we skip if it’s a chapter heading)
    if current_line_stripped in chapter_map:
        return False

    # 3) Check if current line is all uppercase (excluding punctuation/spaces)
    alpha_only = "".join(ch for ch in current_line_stripped if ch.isalpha())
    if not alpha_only:
        return False  # no alphabetic chars, so can't check uppercase

    return alpha_only.isupper()
def parse_txt_file_to_json(txt_file_path: str):
    chapters_data = []
    current_chapter = None
    
    chapter_intro_lines = []
    current_poem_title = None
    poem_text_blocks = []  # list of poem blocks for the current poem title
    
    def start_new_chapter(chapter_line):
        nonlocal current_chapter, chapter_intro_lines, poem_text_blocks, current_poem_title
        
        # Finalize previous chapter if it exists
        if current_chapter is not None:
            # Store last poem if we have one
            store_current_poem_if_any(current_chapter, current_poem_title, poem_text_blocks)
            # Add chapter intro and push to chapters_data
            current_chapter["chapter_intro"] = "\n".join(chapter_intro_lines).strip()
            chapters_data.append(current_chapter)
        
        # Create new chapter structure
        full_title = chapter_map[chapter_line.strip()]
        current_chapter = {
            "chapter_title": full_title,
            "chapter_intro": "",
            "poems": []
        }
        
        # Reset chapter-level buffers
        chapter_intro_lines.clear()
        current_poem_title = None
        poem_text_blocks.clear()
    
    def start_new_poem(poem_title: str):
        nonlocal current_poem_title, poem_text_blocks, current_chapter
        # Before starting a new poem, store the previous poem if it exists
        store_current_poem_if_any(current_chapter, current_poem_title, poem_text_blocks)
        
        current_poem_title = poem_title.strip()
        poem_text_blocks.clear()
    
    def store_current_poem_if_any(chapter_dict, poem_title, text_blocks):
        """
        Each block in text_blocks is a separate poem. 
        If the same poem title has multiple blocks (due to blank lines),
        store them as separate entries with the same title.
        """
        if poem_title and text_blocks:
            for block in text_blocks:
                clean_block = block.strip()
                if clean_block:
                    chapter_dict["poems"].append({
                        "title": poem_title,
                        "poem": clean_block
                    })
            text_blocks.clear()
    
    # Read file
    with open(txt_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    current_block_lines = []
    prev_line = ""  # We'll track the previously read line for is_poem_title
    
    for line in lines:
        stripped_line = line.strip()
        
        # 1) Check if new chapter
        if is_chapter_line(stripped_line):
            start_new_chapter(stripped_line)
            # Reset block lines
            current_block_lines.clear()
        else:
            # 2) If we have a chapter but no poem title yet, we might be in chapter intro
            if current_chapter and (current_poem_title is None):
                # Check if it's a poem title
                if is_poem_title(line, prev_line, chapter_map):
                    start_new_poem(stripped_line)
                else:
                    # Still part of chapter intro
                    chapter_intro_lines.append(line)
            # 3) If we already have a poem title, we're reading poem lines or detecting next poem
            elif current_chapter and current_poem_title:
                # Blank line => end of current poem block
                if stripped_line == "":
                    if current_block_lines:
                        poem_text_blocks.append("\n".join(current_block_lines))
                        current_block_lines.clear()
                    # If it's consecutive blank lines, we just keep ignoring
                # If it's a new poem title
                elif is_poem_title(line, prev_line, chapter_map):
                    # Store leftover block of the old poem
                    if current_block_lines:
                        poem_text_blocks.append("\n".join(current_block_lines))
                        current_block_lines.clear()
                    # Finalize the poem
                    store_current_poem_if_any(current_chapter, current_poem_title, poem_text_blocks)
                    # Start a new poem
                    start_new_poem(stripped_line)
                else:
                    # It's poem content
                    current_block_lines.append(line)
        
        prev_line = line  # update previous line pointer
    
    # End of file: finalize any remaining poem/chapter
    if current_chapter:
        # If there's an unfinished poem block, store it
        if current_block_lines:
            poem_text_blocks.append("\n".join(current_block_lines))
        store_current_poem_if_any(current_chapter, current_poem_title, poem_text_blocks)
        
        # Wrap up the chapter
        current_chapter["chapter_intro"] = "\n".join(chapter_intro_lines).strip()
        chapters_data.append(current_chapter)
    
    return chapters_data

def poem_length_distribution(chapters_data):
    # Load the JSON data
    with open(chapters_data, "r", encoding="utf-8") as file:
        json_file_path = json.load(file)

    poem_lengths = []

    for chapter in json_file_path:
        for poem_obj in chapter["poems"]:
            poem_text = poem_obj["poem"]
            # Split on whitespace to count words
            word_count = len(poem_text.split())
            poem_lengths.append(word_count)
    
    # Basic statistics
    print(f"Number of poems: {len(poem_lengths)}")
    print(f"Min length: {min(poem_lengths)} words")
    print(f"Max length: {max(poem_lengths)} words")
    print(f"Average length: {sum(poem_lengths)/len(poem_lengths):.2f} words")
    
    # Plot a histogram and get bin data
    n, bins, _ = plt.hist(poem_lengths, bins=20, edgecolor='black')
    plt.title("Poem Word Count Distribution")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")

    # Print bin counts
    for i in range(len(n)):
        print(f"Bin {i + 1}: {n[i]} counts, Range: {bins[i]} - {bins[i+1]}")

    # Save the plot
    plt.savefig("poem_word_count_distribution.png", dpi=300, bbox_inches='tight')
    print("Histogram saved as 'poem_word_count_distribution.png'")

def remove_long_poems(input_file, output_file, max_words=500):
    """
    Removes poems that have more than a specified number of words from a dataset.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to save the filtered JSON file.
        max_words (int): Maximum word count allowed for poems.
    """
    with open(input_file, "r", encoding="utf-8") as infile:
        chapters_data = json.load(infile)

    filtered_chapters = []

    for chapter in chapters_data:
        filtered_poems = []
        for poem in chapter.get("poems", []):
            # Count the words in the poem
            word_count = len(poem["poem"].split())
            if word_count <= max_words:
                filtered_poems.append(poem)
        
        if filtered_poems:  # Only include chapters with remaining poems
            chapter["poems"] = filtered_poems
            filtered_chapters.append(chapter)

    # Save the filtered data to a new JSON file
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(filtered_chapters, outfile, ensure_ascii=False, indent=4)

    print(f"Filtered data saved to '{output_file}'. Poems longer than {max_words} words have been removed.")



def main():
    """
    Main function to choose the operation to perform.
    """
    
    # pdf_to_txt("coleman-barks-the-essential-rumi.pdf", "essential_rumi.txt")    
    # remove_page_numbers("essential_rumi.txt", "essential_rumi_removed_pageno.txt")
    # process_blank_lines("essential_rumi_removed_pageno.txt", "essential_rumi_refined.txt")

    # txt_file_path = "essential_rumi_refined.txt"  # your plain text
    # data = parse_txt_file_to_json(txt_file_path)
    
    # with open("rumi_poems.json", "w", encoding="utf-8") as out_f:
    #     json.dump(data, out_f, ensure_ascii=False, indent=2)

    # poem_length_distribution("rumi_poems.json")

    remove_long_poems("rumi_poems.json", "filtered_rumi_poems.json")


# Run the main function
if __name__ == "__main__":
    main()

