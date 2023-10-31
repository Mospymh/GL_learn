# -*- coding: utf-8 -*-

import configparser
import io
import os
import pickle
from pathlib import Path
from typing import Any
import pandas as pd
from openpyxl import load_workbook
import zipfile


class FileWriter:
    @staticmethod
    def get_excel_writer(file_name: str, append: bool = True) -> pd.ExcelWriter:
        try:
            if not append:
                writer = pd.ExcelWriter(file_name, engine='openpyxl')
                return writer
            book = load_workbook(file_name)  # existed
            writer = pd.ExcelWriter(file_name, engine='openpyxl')
            writer.book = book
            writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        except:
            writer = pd.ExcelWriter(file_name, engine='openpyxl')
        return writer

    @staticmethod
    def write_file_content(file_name: str, content: str) -> None:
        if '../' in file_name:
            dir_path = os.path.dirname(__file__)
            cfg_path = os.path.join(dir_path, file_name)
        else:
            cfg_path = file_name
        with io.open(cfg_path, "a", encoding="utf-8") as f:
            f.write(content)
        return

    @staticmethod
    def write_file_property(file_name: str, section_name: str, key: str, value: str) -> None:
        if '../' in file_name:
            dir_path = os.path.dirname(__file__)
            cfg_path = os.path.join(dir_path, file_name)
        else:
            cfg_path = file_name
        config = configparser.ConfigParser()
        config.read(cfg_path, encoding="utf-8")
        if section_name not in config:
            config.add_section(section_name)
        config.set(section_name, key, value)
        with io.open(cfg_path, 'w', encoding="utf-8") as configfile:
            config.write(configfile)
        return

    @staticmethod
    def clear_file(file_name: str) -> None:
        if '../' in file_name:
            dir_path = os.path.dirname(__file__)
            cfg_path = os.path.join(dir_path, file_name)
        else:
            cfg_path = file_name
        io.open(cfg_path, 'w', encoding="utf-8").close()
        return

    @staticmethod
    def df_to_hdf(df: pd.DataFrame, file_name: str) -> None:
        if Path(file_name).is_file():
            os.remove(file_name)
        df.to_hdf(file_name, 'table')
        return

    @staticmethod
    def object_dump_to_file(obj: Any, file_name: str) -> None:
        if Path(file_name).is_file():
            os.remove(file_name)
        pickle.dump(obj, open(file_name, "wb"), protocol=4)
        return

    @staticmethod
    def zip_directory_file(path_name: str, file_name: str) -> None:
        zip_file = zipfile.ZipFile(file_name, 'w', zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(path_name):
            for file in files:
                zip_file.write(os.path.join(root, file), file, zipfile.ZIP_DEFLATED)
        zip_file.close()
        return
