# By default, SQlite does not allow whitespace in either table/column name, unless we use double quotes to quote the name.
# Therefore, to ease the evaluation, we replace all the whitespace in table/column name with underscore.
# Also, we normalize the column names into lowercase.

import argparse 
import os
import json 
import logging 
from tqdm import tqdm

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_token_with_typo(query_tok):
    """
    There is a typo in the original dataset.
    Specifically, they use "column_name" field in SQL instead of "column_name_original" field.
    These include:
    - "chi_tiết khu nhà" -> "chi_tiết_về_khu_nhà"
    - "ngày ngày đăng_ký" -> "ngày_đăng_ký"

    Return None if the given token is not a typo.
    """

    query_tok_replaced = None 

    if "." in query_tok and query_tok != ".":
        splitted = query_tok.split(".")
        if len(splitted) == 2:
            if splitted[1] == "chi_tiết khu nhà":
                query_tok_replaced = splitted[0] + "." + "chi_tiết_về_khu_nhà"
            elif splitted[1] == "ngày ngày đăng_ký":
                query_tok_replaced = splitted[0] + "." + "ngày_đăng_ký"
    else:
        if query_tok == "chi_tiết khu nhà":
            query_tok_replaced = "chi_tiết_về_khu_nhà"
        elif query_tok == "ngày ngày đăng_ký":
            query_tok_replaced = "ngày_đăng_ký"

    return query_tok_replaced



def main(args):
    logger.info("Replacing whitespace in table/column names with underscore.")

    with open(args.table_path, "r") as f:
        table_info = json.load(f)
    
    table_info_renamed = []
    
    for table in tqdm(table_info, desc="Replacing tables.json..."):
        column_names_info = table["column_names"]
        column_names_original_info = table["column_names_original"]
        table_names = table["table_names"]
        table_names_original = table["table_names_original"]

        column_names_info_renamed = []
        for column_info in column_names_info:
            tidx = column_info[0]
            column_name = column_info[1]
            column_name_renamed = column_name.replace(" ", "_").lower()
            # Some column name in VSpider starts with number. e.g) 100_mét, 200_mét..
            # As SQLite does not recognize column names that start with number without " ", we add "_" prefix to such column name.
            if column_name_renamed[0].isdigit():
                column_name_renamed = "_" + column_name_renamed

            column_names_info_renamed.append([tidx, column_name_renamed])
        
        column_names_original_info_renamed = []
        for column_info in column_names_original_info:
            tidx = column_info[0]
            column_name = column_info[1]
            column_name_renamed = column_name.replace(" ", "_").lower()
            if column_name_renamed[0].isdigit():
                column_name_renamed = "_" + column_name_renamed
        
            column_names_original_info_renamed.append([tidx, column_name_renamed])
        
        table_names_renamed = []
        for table_name in table_names:
            table_name_renamed = table_name.replace(" ", "_")
            table_names_renamed.append(table_name_renamed)
        
        table_names_original_renamed = []
        for table_name in table_names_original:
            table_name_renamed = table_name.replace(" ", "_")
            table_names_original_renamed.append(table_name_renamed)
        
        table_info_renamed.append(
            {
                "column_names": column_names_info_renamed,
                "column_names_original": column_names_original_info_renamed,
                "column_types": table["column_types"],
                "db_id" : table["db_id"],
                "foreign_keys": table["foreign_keys"],
                "primary_keys": table["primary_keys"],
                "table_names": table_names_renamed,
                "table_names_original": table_names_original_renamed,
            }
        )
    
    converted_table_save_path = args.table_path.replace(".json", "_converted.json")
    with open(converted_table_save_path, "w") as f:
        json.dump(table_info_renamed, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Saved converted table information to {converted_table_save_path}.")
    logger.info("Finish replacing whitespace in table/column names with underscore.")
    
    logger.info("Replacing whitespace in dataset with underscore.")

    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)
    
    table_info_renamed_dict = {}
    for table in table_info_renamed:
        table_info_renamed_dict[table["db_id"]] = table
    
    dataset_renamed = []
    for data in tqdm(dataset, desc="Replacing dataset..."):
        # Replace the query_toks and query accordingly.
        db_id = data["db_id"]
        table_info = table_info_renamed_dict[db_id]
        all_tables_and_columns = table_info["table_names_original"] + [column_info[1] for column_info in table_info["column_names_original"]]
        query_toks = data["query_toks"]
        query_toks_no_value = data["query_toks_no_value"]
        query_toks_replaced = []
        for query_tok in query_toks:
            
            # First, check if this token is a typo.
            query_tok_replaced = process_token_with_typo(query_tok)
            if query_tok_replaced is not None:
                query_toks_replaced.append(query_tok_replaced)
                continue
            
            if "." in query_tok and query_tok != ".":
                splitted = query_tok.split(".")
                if len(splitted) == 2:
                    if splitted[1][0].isdigit():
                        if "_" + splitted[1].replace(" ", "_").lower() in all_tables_and_columns:
                            query_tok_replaced = splitted[0] + "." + "_" + splitted[1].replace(" ", "_").lower()
                        else:
                            query_tok_replaced = query_tok
                    else:
                        if splitted[1].replace(" ", "_").lower() in all_tables_and_columns:
                            query_tok_replaced = splitted[0] + "." + splitted[1].replace(" ", "_").lower()
                        else:
                            query_tok_replaced = query_tok
            else:
                if query_tok[0].isdigit():
                    if "_" + query_tok.replace(" ", "_").lower() in all_tables_and_columns:
                        query_tok_replaced = "_" + query_tok.replace(" ", "_").lower()
                    else:
                        query_tok_replaced = query_tok
                else:
                    if query_tok.replace(" ", "_").lower() in all_tables_and_columns:
                        query_tok_replaced = query_tok.replace(" ", "_").lower()
                    else:
                        query_tok_replaced = query_tok

            if query_tok_replaced is None:
                query_tok_replaced = query_tok

            query_toks_replaced.append(query_tok_replaced)
        
        query_replaced = " ".join(query_toks_replaced)
        data["query"] = query_replaced
        data["query_toks"] = query_toks_replaced

        query_toks_no_value_replaced = []
        for query_tok in query_toks_no_value:

            # First, check if this token is a typo.
            query_tok_replaced = process_token_with_typo(query_tok)
            if query_tok_replaced is not None:
                query_toks_no_value_replaced.append(query_tok_replaced)
                continue

            if "." in query_tok and query_tok != ".":
                splitted = query_tok.split(".")
                if len(splitted) == 2:
                    if splitted[1][0].isdigit():
                        if "_" + splitted[1].replace(" ", "_").lower() in all_tables_and_columns:
                            query_tok_replaced = splitted[0] + "." + "_" + splitted[1].replace(" ", "_").lower()
                        else:
                            query_tok_replaced = query_tok
                    else:
                        if splitted[1].replace(" ", "_").lower() in all_tables_and_columns:
                            query_tok_replaced = splitted[0] + "." + splitted[1].replace(" ", "_").lower()
                        else:
                            query_tok_replaced = query_tok
            else:
                if query_tok[0].isdigit():
                    if "_" + query_tok.replace(" ", "_").lower() in all_tables_and_columns:
                        query_tok_replaced = "_" + query_tok.replace(" ", "_").lower()
                    else:
                        query_tok_replaced = query_tok
                else:
                    if query_tok.replace(" ", "_").lower() in all_tables_and_columns:
                        query_tok_replaced = query_tok.replace(" ", "_").lower()
                    else:
                        query_tok_replaced = query_tok

            if query_tok_replaced is None:
                query_tok_replaced = query_tok
            query_toks_no_value_replaced.append(query_tok_replaced)

        data["query_toks_no_value"] = query_toks_no_value_replaced
        dataset_renamed.append(data)
    
    converted_dataset_save_path = args.dataset_path.replace(".json", "_converted.json")
    with open(converted_dataset_save_path, "w") as f:
        json.dump(dataset_renamed, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Saved converted dataset to {converted_dataset_save_path}.")
    logger.info("Finish replacing whitespace in dataset with underscore.")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table_path", type=str, default="./data/Vspider/tables.json")
    parser.add_argument("--dataset_path", type=str, default="./data/Vspider/dev.json")

    args = parser.parse_args()
    main(args)