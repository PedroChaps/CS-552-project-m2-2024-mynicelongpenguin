import json


def format_epfl_preference_pairs(input_file, output_file):
    # Load the JSON data from the file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Prepare to write to the output file
    with open(output_file, 'w') as outfile:
        for entry in data:
            prompt = entry['question_complete']
            # Iterate over each preference pair
            for preference in entry['preference']:
                # Determine which answer is chosen based on the 'overall' rating
                if preference['overall'] == 'A':
                    chosen = preference['A']
                    rejected = preference['B']
                else:
                    chosen = preference['B']
                    rejected = preference['A']

                # Format the output
                output_dict = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                }
                
                # Write the formatted dictionary to the file as a JSON string
                json_line = json.dumps(output_dict)
                outfile.write(json_line + '\n')


if __name__ == '__main__':
    input_path = '../data/M1_preference_data_15052024.json'
    output_path = '../data/dpo_formatted_epfl_preference_data.jsonl'
    
    # Call the function to format the data
    format_epfl_preference_pairs(input_path, output_path)