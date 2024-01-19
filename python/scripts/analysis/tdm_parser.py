import logging
import os
import re
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import click
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

FLOAT_REGEX = '[-+]?[0-9]*[.]?[0-9]+'
INT_REGEX = '\d+'
ANY_REGEX = '.*'
ISO_DATE_REGEX = '\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d\.\d+'


def profile(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"{func.__name__} took {elapsed_time:.6f} seconds to execute.")
        return result

    return wrapper


class TDMRegexPatterns(Enum):
    # META Data Regex Patterns
    BEAM_ID = re.compile(fr"COMMENT Data from Beam\s({INT_REGEX})")
    BEAM_NOISE = re.compile(fr"COMMENT Beam Noise\s({FLOAT_REGEX})")
    INT_TIME = re.compile(fr"INTEGRATION_INTERVAL\s=\s({FLOAT_REGEX})")
    TARGET = re.compile(fr"PARTICIPANT_2\s=\s({ANY_REGEX})")

    def __str__(self):
        return f"{self.name}: {self.value.pattern}"


def find_files_by_extension(directory, extension):
    path = Path(directory)

    # Use rglob to recursively search for files with the specified extension
    files = path.rglob(f"*.{extension}")

    # Return a list of Path objects representing the file paths
    return list(files)


def read_file(file_path: Path):
    with file_path.open('r') as file:
        data = file.read()

    return data


def extract_data(data, start_marker, end_marker):
    pattern = re.compile(f'{re.escape(start_marker)}(.*?){re.escape(end_marker)}', re.DOTALL)
    matches = pattern.findall(data)

    return matches


def regex_match(regex_pattern: TDMRegexPatterns, input_string: str, group_index: int = 1):
    match = regex_pattern.value.search(input_string)

    if match.group(group_index) is not None:
        return match.group(group_index)
    else:
        logger.warning(f"Could not match regex pattern {regex_pattern} for group {group_index}")


def regex_match_all(regex_pattern: TDMRegexPatterns, input_string: str, group_index: int = 1):
    # Find all instances of the pattern in the text
    matches = regex_pattern.value.findall(input_string)

    if matches is not None:
        return [match[group_index - 1] for match in matches]
    else:
        logger.warning(f"Could not match regex pattern {regex_pattern} for group {group_index}")


def detection_extract_value(detection_data, entry_index=-1):
    return [entry.split()[entry_index] for entry in detection_data]


def process_detection_data(tdm_filepath, meta_records, extracted_detection_data):
    records = []
    for record_id, beam_data in enumerate(extracted_detection_data):
        # Split records
        a = beam_data.splitlines()

        date = detection_extract_value(a[2::6], 2)
        snr = detection_extract_value(a[4::6])
        tx = detection_extract_value(a[5::6])
        rx = detection_extract_value(a[6::6])

        n = len(date)

        az = [a[2].split()[3]] * n
        el = [a[2].split()[3]] * n

        file_id = [meta_records[record_id][0]] * n
        tdm_date = [meta_records[record_id][1]] * n
        beam_id = [meta_records[record_id][2]] * n
        beam_noise = [meta_records[record_id][3]] * n
        int_time = [meta_records[record_id][4]] * n
        target_name = [meta_records[record_id][5]] * n

        filepath = [tdm_filepath.stem] * n

        records.extend(
            zip(file_id, tdm_date, filepath, date, az, el, snr, tx, rx, beam_id, beam_noise, int_time, target_name)
        )

    column_dtypes = {
        'FILE_ID': int,
        'TDM Date': 'datetime64[ns]',
        'Filename': str,
        'Timestamp': 'datetime64[ns]',
        'AZ': float,
        'EL': float,
        'SNR': float,
        'TX_FREQ': float,
        'RX_FREQ': float,
        'BEAM_ID': int,
        'BEAM_NOISE': float,
        'INT_TIME': float,
        'TARGET': 'str'
    }

    df = pd.DataFrame(records, columns=list(column_dtypes.keys()))
    df = df.astype(column_dtypes)

    df['Doppler'] = df['RX_FREQ'] - df['TX_FREQ']

    return df


@profile
def parse_tdm(file_id: int, tdm_filepath: Path) -> pd.DataFrame:
    raw_data = read_file(tdm_filepath)

    tdm_date = datetime.strptime(tdm_filepath.stem.split('_')[3], '%Y%m%dT%H%M%S')

    # Extract the META data from the TDM
    extracted_meta_data = extract_data(raw_data, 'META_START', 'META_STOP')

    meta_records = []
    for record_id, beam_meta in enumerate(extracted_meta_data):
        meta_records.append([
            file_id,
            tdm_date,
            regex_match(TDMRegexPatterns.BEAM_ID, beam_meta),  # Beam ID
            regex_match(TDMRegexPatterns.BEAM_NOISE, beam_meta),  # Beam Noise
            regex_match(TDMRegexPatterns.INT_TIME, beam_meta),  # Interval Time
            regex_match(TDMRegexPatterns.TARGET, beam_meta)])  # Target Name

    # Extract the detection data from the TDM
    extracted_detection_data = extract_data(raw_data, 'DATA_START', 'DATA_STOP')

    t1 = time.time()
    df = process_detection_data(tdm_filepath, meta_records, extracted_detection_data)
    logger.debug(f'{tdm_filepath.stem} processed in : {time.time() - t1:.3f} seconds')

    return df


def summarise_detection(df: pd.DataFrame):
    min_max_median = ['mean', 'min', 'max']

    # Define aggregation functions for each column
    agg_func = {
        'FILE_ID': 'first',
        'TDM Date': 'first',
        'Timestamp': min_max_median,
        'AZ': 'first',
        'EL': 'first',
        'SNR': min_max_median,
        'TX_FREQ': min_max_median,
        'RX_FREQ': min_max_median,
        'BEAM_NOISE': 'first',
        'INT_TIME': 'first',
        'TARGET': 'first'
    }

    # Group by 'Filename and Beam ID' and apply aggregation functions
    grouped_df = df.groupby(['Filename', 'BEAM_ID']).agg(agg_func).reset_index()

    return grouped_df


def export_df(df, output_file):
    if not output_file:
        return

    export_type = output_file.split('.')[1]
    if export_type == 'csv':
        df.to_csv(f'{output_file}')
    elif export_type == 'xlsx':
        df.to_excel(f'{output_file}')
    elif export_type == 'h5':
        df.to_hdf(f'{output_file}', key='data', mode='w')
    elif export_type == 'pkl':
        df.to_pickle(f'{output_file}')

    click.echo(f'Data exported to {output_file}.')


def config_log():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    return logging.getLogger(__name__)


def parse_tdms(tdm_files, export_filepath):
    # Process the tdm files and summarise the data
    tdm_summarised_data = []
    t0 = time.time()
    for file_id, tdm_filepath in enumerate(tdm_files):
        tdm_data_df = parse_tdm(file_id, tdm_filepath)

        tdm_summarised_data.append(summarise_detection(tdm_data_df))

        logger.info(f'Processed {file_id + 1}/{len(tdm_files)} TDMs. Time elapsed: {time.time() - t0:.3f} seconds')

    logger.info(f'{len(tdm_files)} TDMs processed in : {time.time() - t0:.3f} seconds')

    # Concatenate DataFrames vertically
    stacked_df = pd.concat(tdm_summarised_data, ignore_index=True)

    # Display the stacked DataFrame
    logger.info(f'{len(tdm_files)} TDMs produced {len(stacked_df)} entries')

    # Export results
    if export_filepath:
        export_df(stacked_df, export_filepath)
    else:
        print(stacked_df)


@click.command()
@click.argument('input_path', type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True))
@click.argument('output_file', type=click.Path(), required=False)
def cli(input_path, output_file):
    """
    This program processes TDM files in a specified directory and summarizes the observation data.
    The program accepts an input path, which can be either a directory containing TDM files
    or a single TDM file.

    Additionally, it provides an optional output path for saving the processed results in
    various formats such as CSV, HDF5, Pickle, or Excel.

    :param input_path: Path to the directory containing TDM files or a single TDM file for processing.
    :param output_file: Optional. Path to save the processed results. If not provided, results will be displayed without saving.


    Usage: python tdm_parser.py --input-path INPUT_PATH [--output-path OUTPUT_PATH]

    """
    if os.path.isdir(input_path):
        tdm_files = find_files_by_extension(input_path, extension='tdm')
    elif os.path.isfile(input_path):
        tdm_files = [input_path]
    else:
        click.echo("Invalid input path. No TDM files found. Please provide a valid file or directory.")
        return

    parse_tdms(tdm_files, output_file)


if __name__ == '__main__':
    logger = config_log()

    cli()
