from datasets import load_from_disk
import re
import random

dataset = load_from_disk("../data/theoremqa_justified_2")

def question_specific_process(question_frags, idx):
    options = question_frags[1:]
    clean_answer = question_frags[0]


    if idx in [25, 87, 94, 136, 182, 215, 391, 491, 741]:
        # last option has some of the rest of the question
        last_opt_split = options[-1].split('.')

        if idx == 136:
            clean_answer += '...' + last_opt_split[-1]
        else:
            clean_answer += '.' + last_opt_split[-1]
        
        if idx in [94, 215]:
            options[-1] = last_opt_split[0] + '.' + last_opt_split[1]
        else:
            options[-1] = last_opt_split[0]
    
    elif idx in [550] :
        last_opt_split = options[-1].split(',')
        clean_answer = clean_answer[:-1] + '.' + last_opt_split[-1]
        options[-1] = last_opt_split[0] + ',' + last_opt_split[1] + ',' + last_opt_split[2]
        

    
    return options, clean_answer
    
    
    
def extract_options_and_clean_text(question, idx):
    question_frags = re.split(r' \([a-d]\)', question)
    options, clean_answer = question_specific_process(question_frags, idx)
    return options, clean_answer

def format_entry(entry, idx):
    answer = entry["Answer"]
    question = entry["Question"]
    justification = entry["Justification"]

    #Assumption: there are only 5 options at most
    if answer in ['(a)', '(b)', '(c)', '(d)', '(e)']:
        # get number of the answer
        choice_nr  = ord(answer[1]) - ord('a')
        options, clean_question = extract_options_and_clean_text(question, idx)

        return {
            "question": clean_question,
            "choices" : options,
            "correct_choice": choice_nr,
            "topic": random.choice(["math", "physics"]),
            "answer": justification
        }

    new_answer = justification + '.' + 'Then, the correct answer is: ' + answer
    return {
        "question": question,
        "answer": new_answer,
        "choices": None,
        "correct_choice": None,
        "topic": random.choice(["math", "physics"]) # just to have a rough idea of how much of each topic we have
        # the assumption is that this dataset is 50 / 50 math / physics
    }

new_dataset = dataset.map(format_entry, with_indices=True)
new_dataset.save_to_disk("../data/theoremqa_justified_formatted")