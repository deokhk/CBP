# To evaluate execution accuracy(or TS) on Vspider dataset,
# we need to construct a database for Vspider dataset.
# This script is used to construct a database for Vspider dataset.
# As Vspider dataset has different column/table names than Spider, we update them accordingly.
# We only care whether db present in the dev set of Vspider dataset converted well.

import os 
import argparse 
import json 
import sqlite3
import logging 
from tqdm import tqdm 
from sqlite3 import OperationalError


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



COLUMN_RENAME_REQUEST = 'ALTER TABLE "{table_name}" RENAME COLUMN "{ori_name}" TO "{new_name}";'
TABLE_RENAME_REQUEST = 'ALTER TABLE "{ori_name}" RENAME TO "{new_name}";'


# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            print("Openning a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path, check_same_thread = False)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor


def main(args):

    logger.info("Loading table information from Spider and Vspider dataset.")

    with open(args.spider_table_info, "r", encoding='utf-8') as f:
        spider_table_info = json.load(f)
    with open(args.vspider_table_info, "r", encoding='utf-8') as f:
        vspider_table_info = json.load(f)
    
    spider_table_info_dict = {}
    vspider_table_info_dict = {}

    for table in spider_table_info:
        spider_table_info_dict[table["db_id"]] = table
    for table in vspider_table_info:
        vspider_table_info_dict[table["db_id"]] = table

    logger.info("Start updating column/table names in Vspider database.")

    db_ids = os.listdir(args.db_path)
    for db_id in tqdm(db_ids, desc="Updating..."):

        if db_id == "cre_Drama_Workshop_Groups":
            # Vspider table info for this database has duplicated table names.
            # As this db does not invole in the dev set, we just ignore it.
            continue
        if db_id == "bike_1" or db_id == "academic" or db_id == "yelp":
            # Vspider table info for this database has duplicate column names in this database.
            # As this db does not invole in the dev set, we just ignore it.
            continue

        single_db_path = os.path.join(args.db_path, db_id, f"{db_id}.sqlite")
        cursor = get_cursor_from_path(single_db_path)
        try:
            spider_table = spider_table_info_dict[db_id]
            vspider_table = vspider_table_info_dict[db_id]
        except KeyError:
            # Some database does not have table information in Spider/Vspider dataset.
            # We just ignore it.
            continue

        spider_tb_names = spider_table["table_names_original"]
        vspider_tb_names = vspider_table["table_names_original"]
        spider_column_names = spider_table["column_names_original"]
        vspider_column_names = vspider_table["column_names_original"]

        # Sanity check 
        assert len(spider_tb_names) == len(vspider_tb_names)
        assert len(spider_column_names) == len(vspider_column_names)
        assert spider_table["primary_keys"] == vspider_table["primary_keys"]
        assert spider_table["foreign_keys"] == vspider_table["foreign_keys"]

        num_tables_in_db = len(spider_tb_names)
        # Make it into 2d list 
        spider_column_names_2d = [[] for _ in range(num_tables_in_db)]
        for column_info in spider_column_names:
            tb_idx  = column_info[0]
            column_name = column_info[1]
            if tb_idx != -1:
                # We ignore "*" field.
                spider_column_names_2d[tb_idx].append(column_name)

        vspider_column_names_2d = [[] for _ in range(num_tables_in_db)]
        for column_info in vspider_column_names:
            tb_idx  = column_info[0]
            column_name = column_info[1]
            if tb_idx != -1:
                # We ignore "*" field.
                vspider_column_names_2d[tb_idx].append(column_name)

        for spider_tb_name, vspider_tb_name in zip(spider_tb_names, vspider_tb_names):
            if "sqlite" in vspider_tb_name or "sqlite" in spider_tb_name:
                # We don't need to rename the table names for sqlite system table.
                # [1] not allowed. [2] not necessary.
                continue
            if spider_tb_name == "Album" and vspider_tb_name == "album":
                # Due to case-insensitivity in SQL3, we cannot alter table name from "Album" to "album".
                # We found that the dev set of Vspider does not use this table, so we just ignore it.
                continue

            if spider_tb_name != vspider_tb_name:
                # Rename the table 
                try:
                    out = cursor.execute(TABLE_RENAME_REQUEST.format(ori_name=spider_tb_name, new_name=vspider_tb_name))
                except OperationalError as e:
                    print(TABLE_RENAME_REQUEST.format(ori_name=spider_tb_name, new_name=vspider_tb_name))
                    print(f"DB id: {db_id}")
                    raise e

        for tb_idx, (spider_column_in_tb, vspider_column_in_tb) in enumerate(zip(spider_column_names_2d, vspider_column_names_2d)):
            # Substitute the column names
            # Anyway we need a vspider table name 
            spider_tb_name = spider_tb_names[tb_idx]
            vspider_tb_name = vspider_tb_names[tb_idx]
            if "sqlite" in vspider_tb_name:
                # We don't need to rename the column names for sqlite system table.
                #  [1] not allowed. [2] not necessary.
                continue
            
            if spider_tb_name == "Album" and vspider_tb_name == "album":
                # Due to case-insensitivity in SQL3,  we just ignore it.
                continue

            for spider_column_name, vspider_column_name in zip(spider_column_in_tb, vspider_column_in_tb):
                try:
                    out = cursor.execute(COLUMN_RENAME_REQUEST.format(table_name=vspider_tb_name, ori_name=spider_column_name, new_name=vspider_column_name))
                except OperationalError as e:
                    print(COLUMN_RENAME_REQUEST.format(table_name=vspider_tb_name, ori_name=spider_column_name, new_name=vspider_column_name))
                    print(f"DB id: {db_id}")
                    raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="./vspider_database/")
    parser.add_argument("--spider_table_info", type=str, default="./data/spider/tables.json")
    parser.add_argument("--vspider_table_info", type=str, default="./data/Vspider/tables.json")

    args = parser.parse_args()
    main(args)

