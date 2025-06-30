import time
import warnings
import outlines
from typing import Literal
import json
from utils import template
from dotenv import load_dotenv
from outlines import generate
from outlines.models import Transformers
from outlines.samplers import greedy
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ################# code validation
class CodeValidationResponse(BaseModel):
    is_valid: bool

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

# Load model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

hf_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize Outlines model wrapper
outlines_model = Transformers(hf_model, tokenizer)

# ################# CHOICE ###############
prompt = template(model=outlines_model, prompt="""Look at this restaurant review and classify its sentiment.
                     Respond only with 'positive' or 'negative':
                    Review: The pizza was delicious, and the service was excellent.""")

sentiment_regex = r'(positive|negative)'
chooser = outlines.generate.choice(
    outlines_model,
    ['positive', 'negative'],
    sampler=greedy()
)

response = chooser(prompt)
print(response)

# ############### Phone Number ##############
phone_prompt = template(model=outlines_model, prompt="""
Extract the phone number from the example,
please use the format: (XXX) XXX-XXXX

206-555-1234

""")

phone_regex = r'\([0-9]{3}\) [0-9]{3}-[0-9]{4}'

phone_generator = outlines.generate.regex(
    outlines_model,
    phone_regex,
    sampler=greedy()
)

print(phone_generator(phone_prompt))

# ############ Email ######################
email_regex = r'[a-zA-Z0-9]{3,10}@[a-z]{4,20}\.com'
email_prompt = template(model=outlines_model, prompt="Give me an email address for someone at amazon")
email_generator = outlines.generate.regex(
    outlines_model,
    email_regex,
    sampler=greedy())
print(email_generator(email_prompt))

# ##################### CSV #################
csv_regex = r'Code,Amount,Cost\n([A-Z]{3},[1]*[0-9],1]*[0-9]\.[0-9]{2}\n){1,3}'
csv_generator = outlines.generate.regex(outlines_model, csv_regex)
csv_out = csv_generator(
    template(model=outlines_model, prompt=
        """Create a CSV file for 2-3 store inventory items.
           Include a column 'Code', 'Amount', and 'Cost'.
        """)
)
from io import StringIO
import pandas as pd
print(pd.read_csv(StringIO(csv_out)))

# ##### HTML Image Tag #############
example = '<img src="large_dinosaur.png" alt="Image of Large Dinosaur">'
img_tag_regex = r'<img src="\w+\.(png|jpg|gif)" alt="[\w ]+">'
import re

print(re.search(img_tag_regex, example)[0])
img_tag_generator = outlines.generate.regex(outlines_model, img_tag_regex)

img_tag = img_tag_generator(
    template(model=outlines_model, prompt=
        """Generate a basic html image tag for the file 'big_fish.png',
        make sure to include an alt tag"""
    ))

print(img_tag)

from IPython.display import HTML, display

display(HTML(img_tag))

############## code validation ##########
# Code snippet to validate
code_to_validate = """
def add(a, b)
return a + b
"""

# Prompt to instruct the model to validate the code
validation_prompt = template(model=outlines_model, prompt=f"""
Act as a Python interpreter. 
Determine if the following code has valid Python syntax. 
Respond in JSON with one field: is_valid (boolean).

Only respond with `false` if the code contains syntax errors.

Code:
{code_to_validate}
""")

# Generate structured JSON response using the schema
validate_code = outlines.generate.json(
    outlines_model,
    CodeValidationResponse,  # Pass schema as positional argument
    sampler=greedy()
)

validation_result = validate_code(validation_prompt)

print("\nCode Validation Result:")
print(validation_result)



