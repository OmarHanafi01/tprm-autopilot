import vertexai
import os
from vertexai.language_models import TextGenerationModel


def generate_answer(question: str,
                    context: list,
                    valid_answers: list = None) -> str:
    vertexai.init(project=os.environ["PROJECT_ID"],
                  location=os.environ["LOCATION"])
    parameters = {
        "temperature": 0.1,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40,
    }
    model = TextGenerationModel.from_pretrained("text-bison@002")
    context = ", ".join(context)
    if valid_answers:
        valid_answer_text = f"""- Answer with either {", ".join(valid_answers)}, followed by your explanation.""" # noqa
    else:
        valid_answer_text = ""
    answer_response = model.predict(
        f"""
        "{question}"

        I want to generate an answer the question above from a security audit assessment for OUR organisation
        Strictly follow the instructions below: 
        {valid_answer_text}
        - The answer should be from the point of view of the organisation
        - Use perfect present continuous tense.
        - Answer everything in plain text with no bold, no italices and no special characters.

        Below contains answers from previous vendor risk assessment questions that the organisation has answered before.
        Use them as context(s) to answer for the question:
        {context}
        """, # noqa
        **parameters,
    )
    return answer_response.text
