import re
import pandas as pd
from dotenv import load_dotenv
from utils import DataFrameHandler
from funcs import generate_answer, embedExistingDocuments


def main() -> None:
    file_paths_dict = {
        "current_assessment_path": "path/to/your/current_assessment", # noqa
        "previous_assessments_path": "path/to/your/previous_assessments_contexts", # noqa
        "output_folder": "data/output"
    }
    CURRENT_ASSESSMENT_QUESTION_COLUMNS = [
        "Section",
        "SubSection",
        "Question",
        "Description",
    ]
    VALID_ANSWERS_LIST = [
        "Partially In Place",
        "Not In Place",
        "Non Applicable",
        "In Place",
    ]

    current_assessment_handler = DataFrameHandler(file_path=file_paths_dict["current_assessment_path"]) # noqa
    current_assessment_df = current_assessment_handler.to_dataframe()

    embedder = embedExistingDocuments()
    embedder.get_assessment_embeddings(
        assessment_path=file_paths_dict["previous_assessments_path"]) # noqa

    csv_data_list = []
    for index, row in current_assessment_df.head(10).iterrows():
        print(index, "/", len(current_assessment_df))
        question_prompt = ""
        row_data_dict = {
            "question": "",
            "answer": "",
            "comment": "",
            "references": "",
        }
        for column in CURRENT_ASSESSMENT_QUESTION_COLUMNS:
            if column_value := row[column]:
                question_prompt += f"{column}: {column_value}"
        question_prompt = question_prompt[:-1]
        similar_contexts = embedder.get_similar_context_from_assessments(
            prompt=question_prompt
        )
        suggested_comment = generate_answer(
            question=question_prompt, context=similar_contexts
        )
        row_data_dict["question"] = row.Question
        for answer in VALID_ANSWERS_LIST:
            if answer.lower() in suggested_comment.lower():
                row_data_dict["Answer"] = answer
                break
        row_data_dict["comment"] = suggested_comment
        if isinstance(similar_contexts, str):
            similar_contexts_text = re.sub(
                r'[^A-Za-z0-9\s!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]',
                "",
                similar_contexts,
            )
        elif isinstance(similar_contexts, list):
            similar_contexts_text = re.sub(
                r'[^A-Za-z0-9\s!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]',
                "",
                "\n\n".join(similar_contexts),
            )
        row_data_dict["references"] = similar_contexts_text
        if isinstance(similar_contexts, list):
            for i, context in enumerate(similar_contexts):
                row_data_dict[f"similarContext{i+1}"] = context
        else:
            raise ValueError("Your contexts are invalid.")
        csv_data_list.append(row_data_dict)

    output_file = (
        f"{file_paths_dict["output_folder"]}/Generated {current_assessment_handler:filename}.csv"  # noqa
    )

    df = pd.DataFrame(csv_data_list)
    df.to_csv(output_file, index=False)

if __name__ == "__main__": # noqa
    load_dotenv()
    main()
