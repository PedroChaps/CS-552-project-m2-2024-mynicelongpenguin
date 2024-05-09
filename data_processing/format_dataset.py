from datasets import load_from_disk
import re
import random

dataset = load_from_disk("../data/theoremqa_justified_2")

def extract_options_and_clean_text(question):
    options = re.findall(r"\([abcde]\)", question)
    options_text = [re.sub(r"\([abcde]\) ", "", option) for option in options]
    cleaned_question = re.sub(r"\([abcde]\) .+?(?=\([abcde]\)|$)", "", question)
    return options_text, cleaned_question
    

def format_entry(entry):
    answer = entry["Answer"]
    question = entry["Question"]
    justification = entry["Justification"]

    # # TOO DIFICULT TO HANDLE
    # if answer in ['(a)', '(b)', '(c)', '(d)', '(e)']:
    #     options, cleaned_question = extract_options_and_clean_text(question)
    #     print(question)

    #     print("Options:", options)
    #     print()
    #     print("Cleaned question:", cleaned_question)
    #     print()
    #     print(80 * "-")
    #     print()
    new_answer = justification + '.' + 'Then, the correct answer is: ' + answer
    return {
        "question": question,
        "answer": new_answer,
        "choices": None,
        "correct_choice": None,
        "topic": random.choice(["math", "physics"]) # just to have a rough idea of how much of each topic we have
        # the assumption is that this dataset is 50 / 50 math / physics
    }

new_dataset = dataset.map(format_entry)

new_dataset.save_to_disk("../data/theoremqa_justified_formatted")