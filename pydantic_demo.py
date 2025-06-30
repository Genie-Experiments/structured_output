import instructor
from instructor.exceptions import InstructorRetryException
import warnings
from openai import OpenAI
from pydantic import BaseModel
from pydantic import Field
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

####################### schemas #######################

class NameYear(BaseModel):
    name: str
    year: int

class Car(BaseModel):
    make: str
    model: str
    year: int

class Person(BaseModel):
    name: str
    age: int
    occupation: str

class Fruit(BaseModel):
    name: str
    color: str
    sweetness_level: int

class SimpleAnimal(BaseModel):
    species: str
    habitat: str
    diet: str

class Country(BaseModel):
    name: str
    capital: str
    population_millions: int

class Book(BaseModel):
    title: str
    author: str
    published_year: int

class Movie(BaseModel):
    title: str
    director: str
    release_year: int

class City(BaseModel):
    name: str
    country: str
    population: int

class Product(BaseModel):
    id: int
    name: str
    price_usd: float

class Complicated(BaseModel):
    a: Literal["cat", "dog", "animal"]
    b: int
    c: bool

# # Test cases
prompts = [
    ("Give me a JSON object with your name and the year you were created.", NameYear),
    ("Create a JSON where the name contains special characters @#$ and year is negative", NameYear),
    ("Generate output with name as empty string and year as 3.14 (should be integer)", NameYear),
    ("Return data with extra field 'timestamp' that shouldn't be present", NameYear),
    ("Generate a JSON object containing your name, the year of your creation, and the organization that developed you.", NameYear),
    ("List your name, creation year, and the programming languages you're proficient in.", NameYear),
    ("Describe a car with make, model, year, and engine type.", Car),
    ("Provide details of a car including make, model, year, and fuel efficiency in km/l.", Car),
    ("List a car's make, model, year, and whether it's electric or gasoline-powered.", Car),
    ("Provide a car's make, model, year, and its safety rating out of 5", Car),
    ("Create a JSON object for a person including name, age, occupation, and nationality.", Person),
    ("List a person's name, age, occupation, and years of experience in their field.", Person),
    ("Provide details of a person with name, age, occupation, and their highest educational qualification.", Person),
    ("Generate a JSON object for a person including name, age, occupation, and marital status.", Person),
    ("Describe a person with name, age, occupation, and their primary language.", Person),
    ("Provide a JSON object for a fruit including name, color, sweetness level, and average weight in grams.", Fruit),
    ("List a fruit's name, color, sweetness level, and its season of availability.", Fruit),
    ("Generate details of a fruit with name, color, sweetness level, and vitamin C content in mg.", Fruit),
    ("Describe a fruit including name, color, sweetness level, and whether it's tropical or temperate.", Fruit),
    ("Provide a fruit's name, color, sweetness level, and common culinary uses.", Fruit),
    ("Create a JSON object for an animal including species, habitat, diet, and average lifespan in years.", SimpleAnimal),
    ("List an animal's species, habitat, diet, and its conservation status.", SimpleAnimal),
    ("Provide details of an animal with species, habitat, diet, and typical group behavior (e.g., solitary, pack).", SimpleAnimal),
    ("Describe an animal including species, habitat, diet, and its primary predators.", SimpleAnimal),
    ("Generate a JSON object for an animal with species, habitat, diet, and whether it's nocturnal or diurnal.", SimpleAnimal),
    ("Provide a JSON object for a country including name, capital, population in millions, and official language.", Country),
    ("List a country's name, capital, population in millions, and its currency.", Country),
    ("Generate details of a country with name, capital, population in millions, and its form of government.", Country),
    ("Describe a country including name, capital, population in millions, and its primary export.", Country),
    ("Provide a country's name, capital, population in millions, and its continent.", Country),
    ("Create a JSON object for a book including title, author, published year, and genre.", Book),
    ("List a book's title, author, published year, and number of pages.", Book),
    ("Provide details of a book with title, author, published year, and its ISBN number.", Book),
    ("Describe a book including title, author, published year, and its target audience", Book),
    ("Generate a JSON object for a book with title, author, published year, and whether it's part of a series.", Book),
    ("Provide a JSON object for a movie including title, director, release year, and genre.", Movie),
    ("List a movie's title, director, release year, and its main actor.", Movie),
    ("Generate details of a movie with title, director, release year, and its duration in minutes.", Movie),
    ("Describe a movie including title, director, release year, and its box office earnings in USD.", Movie),
    ("Provide a movie's title, director, release year, and its IMDb rating.", Movie),
    ("Create a JSON object for a city including name, country, population, and area in square kilometers.", City),
    ("List a city's name, country, population, and its founding year.", City),
    ("Provide details of a city with name, country, population, and its primary language.", City),
    ("Describe a city including name, country, population, and its major industries.", City),
    ("Generate a JSON object for a city with name, country, population, and its average annual temperature in Celsius.", City),
    ("Provide a JSON object for a product including id, name, price in USD, and category.", Product),
    ("List a product's id, name, price in USD, and its manufacturer.", NameYear),
    ("Generate details of a product with id, name, price in USD, and its stock availability.", NameYear),
    ("Describe a product including id, name, price in USD, and its warranty period in months.", NameYear),
    ("Provide a product's id, name, price in USD, and its average customer rating out of 5.", NameYear),
    ("Create a JSON object with your name, year of creation, and your primary function or role.", NameYear),
    ("Provide your name, the year you were created, and the version number of your current iteration.", NameYear),
    ("Produce output where the name is 12345 (should be string) and year is 'two thousand'", NameYear),
    ("Create JSON with unicode name 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 and year 0x10 (hexadecimal)", NameYear),
    ("Generate car data where make is 123 and model is True (should be strings)", Car),
    ("Return JSON where model contains newlines and tabs", Car),
    ("Produce car data with year 'twenty twenty three", Car),
    ("Generate output where make is null and model is extremely long string", Car),
    ("Return a JSON object describing a car with make, model, and year.", Car),
    ("Create person data with age 'thirty-five'", Person),
    ("Generate output where occupation is a list ['teacher', 'writer'] instead of string", Person),
    ("Return JSON with nested address object that shouldn't exist", Person),
    ("Output a JSON object listing the name, age, and occupation of a person.", Person),
    ("Create a JSON object for a fruit with name, color, and sweetness level.", Fruit),
    ("Generate fruit data where sweetness_level is 'very sweet' instead of integer", Fruit),
    ("Create JSON with color as RGB array [255,0,0] instead of string", Fruit),
    ("Return output with name in ALL CAPS and negative sweetness", Fruit),
    ("Produce fruit data with extra 'expiry_date' field", Fruit),
    ("Generate output where color is 7 (should be string)", Fruit),
    ("Create animal data where diet is a dictionary {main: 'plants', occasional: 'meat'}", SimpleAnimal),
    ("Generate output with habitat as integer 42 instead of string", SimpleAnimal),
    ("Return JSON where species contains XML tags <species>Wolf</species>", SimpleAnimal),
    ("Produce animal data with missing 'diet' field", SimpleAnimal),
    ("Generate output with all fields set to null", SimpleAnimal),
    ("Give me a JSON object for an animal with species, habitat, and diet.", SimpleAnimal),
    ("Describe a country in JSON with name, capital, and population (in millions).", Country),
    ("Output a JSON object for a book with title, author, and published year.", Book),
    ("Describe a movie in JSON with title, director, and release year.", Movie),
    ("Create movie data where release_year is in future (2100)", Movie),
    ("Generate output with director as list ['Director1', 'Director2']", Movie),
    ("Return JSON where title is Unicode 𝕋𝕙𝕖 𝕄𝕒𝕥𝕣𝕚𝕩", Movie),
    ("Produce movie data with release_year as string '2000'", Movie),
    ("Generate output with extra 'sequels' field", Movie),
    ("Give me a JSON object for a city with name, country, and population.", City),
    ("Create city data where population is 'about one million'", City),
    ("Generate output with country as country code 'US' instead of full name", City),
    ("Produce city data with population as scientific notation 1e6", City),
    ("Generate output with missing 'country' field", City),
    ("Create product data where price_usd is '$19.99' with dollar sign", Product),
    ("Generate output with id as string '123' instead of integer", Product),
    ("Return JSON where name contains HTML <b>Premium</b>", Product),
    ("Produce product data with negative price", Product),
    ("Generate output with extra 'discount' field", Product),
    ("Describe a product with ID, name, and price in USD as JSON.", Product),
    ("Create data where a is 'CAT' (uppercase not in Literal)", Complicated),
    ("Generate output where b is True instead of integer", Complicated),
    ("Return JSON where c is 'yes' instead of boolean", Complicated),
    ("Produce data with missing 'a' field", Complicated),
    ("Generate output where all values are null", Complicated),
    ("Give me a JSON object for values in a, b and c", Complicated)
]

def generate_responses(response_model, user_prompt, system_prompt=None):
    try:
        system_content = system_prompt if system_prompt else ""
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ]
        )

        raw = completion.choices[0].message.content.strip()
        print("*************", raw)

        result = response_model.model_validate_json(raw)

        return {
            "status": "Success",
            "model": response_model.__name__,
            "prompt": user_prompt,
            "raw_response": raw,
            "parsed_result": result.dict()
        }

    except Exception as e:
        return {
            "status": "Failure",
            "model": response_model.__name__,
            "prompt": user_prompt,
            "raw_response": raw if 'raw' in locals() else None,
            "error": str(e)
        }

results_log = []
success_count = 0
failure_count = 0

for user_prompt, model in prompts:
    result = generate_responses(model, user_prompt)
    results_log.append(result)

    if result["status"] == "Success":
        success_count += 1
        print(f"[SUCCESS] {model.__name__} - Parsed correctly.")
    else:
        failure_count += 1
        print(f"[FAILURE] {model.__name__} - See raw response below:")
        print(result["raw_response"])
        print(f"Error: {result['error']}\n")

# Calculate statistics
total_cases = len(prompts)
success_rate = (success_count / total_cases) * 100
failure_rate = (failure_count / total_cases) * 100

# Print summary
print("\n=== Test Summary ===")
print(f"Total Test Cases: {total_cases}")
print(f"Successes: {success_count} ({success_rate:.2f}%)")
print(f"Failures: {failure_count} ({failure_rate:.2f}%)")

# Breakdown by model
print("\n=== Model Performance Breakdown ===")
model_stats = {}
for user_prompt, model in prompts:
    model_name = model.__name__
    if model_name not in model_stats:
        model_stats[model_name] = {"success": 0, "failure": 0}

for result in results_log:
    model_name = result["model"]
    if result["status"] == "Success":
        model_stats[model_name]["success"] += 1
    else:
        model_stats[model_name]["failure"] += 1

for model, stats in model_stats.items():
    total = stats["success"] + stats["failure"]
    success_pct = (stats["success"] / total) * 100 if total > 0 else 0
    print(f"{model}: {stats['success']}/{total} ({success_pct:.2f}%) successful")

# Save results
import json
with open("pydantic_structured_output_test_results.json", "w") as f:
    json.dump({
        "summary": {
            "total_cases": total_cases,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_rate,
            "failure_rate": failure_rate
        },
        "model_breakdown": model_stats,
        "detailed_results": results_log
    }, f, indent=4)

print("\nTest run complete. Detailed results saved to pydantic_structured_output_test_results.json")