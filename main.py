# main.py
import argparse

import instructor_demo

import warnings
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, List, Literal
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()  # This loads the variables from the .env file

KEY = os.getenv("KEY")

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize Groq client

client = Groq(api_key=KEY)



################################

# Social Media Mention Structure

################################

class Mention(BaseModel):
    product: Literal['app', 'website', 'not_applicable']
    sentiment: Literal['positive', 'negative', 'neutral']
    needs_response: bool
    response: Optional[str]
    support_ticket_description: Optional[str]

mentions = [
    "@techcorp your app is amazing! The new design is perfect",
    "@techcorp website is down again, please fix!",
    "hey @techcorp you're so evil",
    "Damn! @techcorp you're killing it out there"
]

def analyze_mention(mention: str, personality: str = "rude") -> Mention:
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": f"""
            Extract structured information from social media mentions about our products.

            Your JSON must strictly match this structure:
            {{
                "product": "app" | "website" | "not_applicable",
                "sentiment": "positive" | "negative" | "neutral",
                "needs_response": true | false,
                "response": string | null,
                "support_ticket_description": string | null
            }}

            Notes:
            - 'product' must be exactly: app, website, or not_applicable.
            - 'needs_response' must be true or false.
            - 'response' should be a string if responding, else null.
            - 'support_ticket_description' should be filled only if technical action is needed.

            Your personality is {personality}.
            Only reply with valid JSON. Do not change the field names.
            """},
            {"role": "user", "content": mention},
        ]
    )
    raw = completion.choices[0].message.content.strip()
    print(type(raw))
    print(raw)

    try:
        # Validate the JSON
        return Mention.model_validate_json(raw)
    except Exception as e:
        print(f"Failed to parse AI response: {raw}")
        raise e

responses = []

for mention in mentions:
    try:
        response = analyze_mention(mention, personality="rude")
        responses.append(response)
    except Exception as e:
        print(f"Error analyzing mention: {mention}")
        print(e)

print(responses)

# class User(BaseModel):
#     name: str
#     age: int
#     email: Optional[str] = None
# completion = client.beta.chat.completions.parse(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Make up a user."},
#     ],
#     response_format=User,
# )
