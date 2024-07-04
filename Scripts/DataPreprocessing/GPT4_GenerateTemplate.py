"""
This code is used to prepare training data for llama3:8b

1. consolidated ITEM7 HTML with: preparing_data()
2. generate training data with: generate_from_10k()
3. save it as input (HTML) and output (JSON): save_output()
    reason why we save it first it is because we can view the outcome that ChatGPT is correct or not
4. push to Hub: Push_Hub()


"""


from openai import OpenAI
import pandas as pd
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import random
import os
from typing import Any, Dict, List, Optional
from huggingface_hub import login
from datasets import Dataset

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

## Please set your OPEN AI KEY HERE
client = OpenAI(api_key=os.environ['OPEN_AI_KEY'])
login(os.environ['HUGGING_FACE_TOKEN'])

class PTrainData():

    def __init__(self, RAW_DATA_PATH=r'..\..\_static\HTML_ITEMS', GPT_OUTPUT_PATH=r'DATA\DataOut\Output'):

        self.RAW_DATA_PATH = RAW_DATA_PATH
        self.system_prompt = """
        You are a helpful assistant designed to output JSON.
        You will be provided with a HTML content that include tables or paragraphs, Please help me extract the financial statistics from those tables or paragraphs as the formatted template.
    
        The formatted template will be:
        {"data": 
        [
        {"Category": "...", "Subcategory": "...","Name": "...", "Date": ..., "Value": ..., "Value_Sign": "", "Unit": "...", "Change_Direction": ""}
        ]
        }
    
        Noted: 
        1. `Category`: The main classification to which a subcategory belongs.
        2. `Subcategory`: A more specific classification within a category.
        3. `Name`: A financial term (e.g., securities, revenue, loan, etc.).
        4. `Unit`: could be "thousands"/"millions"/"billions" (if it is an amount, and mentioned in the table/paragraph) or "percentage"/"%" (if changes) or "" (if no mentioned). please set `Unit` to "" if the provided input does not explicitly include the words "thousands", "millions", "billions", "percentage" or "%".  For example, Apple's net income increased by 10 millions, then it is an "millions", Apple's net income increased by 10,000,000, then it is an "".
        5. `Change_Direction`: can only be "increase" or "decrease" or "", please set Change_Direction to "" if the provided input does not explicitly include the words "increase" or "decrease". For example, Apple's net income increased by 10 millions, then it is an "increase".
        6. `Value_Sign`: can only be "positive" or "negative". For example, 100 is "positive", (100) and -100 are "negative" sign
    
        1. Anything that you can not find it, just fill it as empty string such as "", but please keep it as the format template.
        2. Please extract all the information from the input provided!!
        """
        self.GPT_OUTPUT_PATH = GPT_OUTPUT_PATH

    def preparing_data(self):

        list_out = []
        main_path = self.RAW_DATA_PATH
        comp_list = os.listdir(main_path)
        for comp in comp_list:
            comp_path = os.path.join(main_path, comp)
            if comp_path.endswith('.rar') or comp_path.endswith('.csv'):
                continue
            else:
                comp_year = os.listdir(comp_path)
                for i in comp_year:
                    item_path = os.path.join(comp_path, i)
                    items_list = os.listdir(item_path)
                    for item in items_list:
                        item_check = item.replace('.html', '')
                        if item_check == "item7":
                            item_spec_path = os.path.join(item_path, item)
                            comp_name = i.split('-')[0]
                            comp_date = i.split('-')[1]
                            df_temp = pd.DataFrame({'Company_Name': [comp_name], 'Date': [comp_date], "Item": item,
                                                    'HTML_PATH': [item_spec_path]})
                            list_out.append(df_temp)
        df_concat = pd.concat(list_out)
        return df_concat

    def generate_from_10k(self, comp: str, date: str,
                          item: str, content: str) -> List[str]:

        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ]
        )
        return [comp, date, item, content, response.choices[0].message.content]

    def convert_json(self, x: str) -> str:
        try:
            out = json.loads(x)
            return out
        except:
            return None

    def save_output(self, output_list: List) -> None:

        df_out = pd.DataFrame(output_list, columns=['Company_Name', 'Date', 'Item', 'HTML_TABLE', 'JSON_OBJ'])
        df_out['JSON_OBJ_formatted'] = df_out.JSON_OBJ.apply(lambda x: self.convert_json(x))
        df_out = df_out[df_out.JSON_OBJ_formatted.apply(lambda x: len(str(x)) > 200)]
        if not os.path.exists(self.GPT_OUTPUT_PATH):
            os.makedirs(self.GPT_OUTPUT_PATH)
        for i, data in enumerate(df_out.JSON_OBJ.apply(lambda x: self.convert_json(x))):
            with open(f'{self.GPT_OUTPUT_PATH}/data_{i}.json', 'w') as f:
                json.dump(data, f, indent=4)

        for i, data in enumerate(df_out.HTML_TABLE):
            with open(f'{self.GPT_OUTPUT_PATH}/data_{i}.html', 'w', encoding='utf-8') as f:
                f.write(data)

    def push_hub(self):

        main_path = self.GPT_OUTPUT_PATH
        list_data = os.listdir(main_path)
        train_list = []
        num = []
        for data in list_data:
            if data.endswith('html'):
                data = data.replace('.html', '')
                data_ = data.split('_')
                num.append(data_[1])

        for i in num:
            json_path = os.path.join(main_path, f"data_{i}.json")
            html_path = os.path.join(main_path, f'data_{i}.html')
            with open(html_path, 'r', encoding='utf-8') as file:
                input = file.read()

            with open(json_path, 'r', encoding='utf-8') as file:
                output = json.load(file)
            dict_temp = {'instruction': self.system_prompt, 'input': input, 'output': output}
            train_list.append(dict_temp)

        dataset_data = Dataset.from_list(train_list)
        dataset_splited = dataset_data.train_test_split(test_size=0.3)
        dataset_splited.push_to_hub(os.environ['HF_DATA_PATH'])



    def run(self):

        ################################################ 1. Prepare Input Data for GPT #####################################
        df = self.preparing_data()

        ################################################ 2. GPT Generating Process #########################################

        output_list = []
        for i, row in tqdm(df.iterrows()):
            print('---------------------------------')
            print(i)
            Company_Name = row['Company_Name']
            Date = row['Date']
            Item = row['Item'].replace('.html', '')
            if Item == 'item7':
                with open(row['HTML_PATH'], "r", encoding='utf-8') as f:
                    content = f.read()

                bsObj = BeautifulSoup(content, 'html.parser')
                tables = bsObj.find_all('table')

                ## Randomly select 5 tables from company and year's 10K Report
                if len(tables) >= 5:
                    tables_random = random.sample(tables, 5)
                else:
                    tables_random = tables
                for table in tqdm(tables_random):
                    output = self.generate_from_10k(Company_Name, Date, Item, str(table))
                    output_list.append(output)
            print('---------------------------------')

        ################################################ 3. Save GPT Output Local #################################################
        self.save_output(output_list)

        ################################################ 4. Push to Hub ###########################################################
        self.push_hub()

if __name__ == '__main__':

    ptraindata = PTrainData()
    ptraindata.run()
