#!/usr/bin/env python3

"""
    Donoho Lab Experiment Management System
"""

import copy
import json
import logging
import os
import random
import time
from datetime import datetime, timezone, timedelta
from math import ceil, floor
from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pandas_gbq
import pandas_gbq.exceptions
from dask.distributed import Client, worker_client, as_completed
from google.cloud.sql.connector import Connector
from google.oauth2 import service_account
from pg8000.dbapi import Connection
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.schema import MetaData

DB_URL = 'sqlite:///data/EMS.db3'
BATCH_SIZE = 4096
NUM_CELLS = 200 * 1000  # 200 rows x 1,000 columns. Slightly less than the values used on FarmShare
logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _touch_db_url(db_url: str):
    db_path = db_url.split('sqlite:///')
    if db_path[0] != db_url:  # If the string was found …
        p = Path(db_path[1])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)


def _write_size_check(df: DataFrame) -> bool:
    t_row, n_col = df.shape
    return t_row * n_col > NUM_CELLS


class Databases(object):

    def __init__(self, table_name: str = None,
                 remote: Engine = None,  # SQLAlchemy based systems
                 credentials: service_account.credentials = None, project_id: str = None):  # Google Big Query.
        self.results = []
        self.last_save = _now()
        if table_name is not None:
            self.table_name = table_name
            _touch_db_url(DB_URL)
            self.local = create_engine(DB_URL, echo=False)
            self.remote = remote
            self.credentials = credentials
            self.project_id = project_id
        else:
            self.table_name = None
            self.local = None
            self.remote = None
            self.credentials = None
            self.project_id = None

    def _push_to_database(self):
        df = pd.concat(self.results)
        df.reset_index(drop=True, inplace=True)
        logger.warning(f'_push_to_database(): Number of DataFrames: {len(self.results)}; ' +
                       f'Length of DataFrames: {sum(len(result) for result in self.results)}\n{df}')
        self.results = []
        chunk_size = ceil(len(df) / 2) + 1 if _write_size_check(df) else len(df) + 1
        # Store locally for durability.
        if self.local is not None:
            try:
                with self.local.connect() as ldb:
                    df.to_sql(self.table_name, ldb, if_exists='append', method='multi', chunksize=chunk_size)
            except SQLAlchemyError as e:
                logger.error("%s", e)
        # Store remotely for flexibility.
        if self.remote is not None:
            try:
                with self.remote.connect() as rdb:
                    df.to_sql(self.table_name, rdb, if_exists='append', method='multi', chunksize=chunk_size)
            except SQLAlchemyError as e:
                logger.error("%s", e)
        if self.credentials is not None:
            try:
                pandas_gbq.to_gbq(df, f'EMS.{self.table_name}',
                                  if_exists='append', chunksize=chunk_size,
                                  progress_bar=False,
                                  credentials=self.credentials)
            except pandas_gbq.exceptions.GenericGBQException as e:
                logger.error("%s", e)
        elif self.project_id is not None:
            try:
                pandas_gbq.to_gbq(df, f'EMS.{self.table_name}',
                                  if_exists='append', chunksize=chunk_size,
                                  progress_bar=False,
                                  project_id=self.project_id)
            except pandas_gbq.exceptions.GenericGBQException as e:
                logger.error("%s", e)
        df = None

    def _df_size_check(self, df: DataFrame) -> bool:
        _, n_col = df.shape
        t_row = sum(len(result) for result in self.results)
        return t_row * n_col > NUM_CELLS

    def push(self, result: DataFrame):
        now = _now()
        self.results.append(result)
        if self._df_size_check(result) or (now - self.last_save) > timedelta(seconds=60.0):
            self._push_to_database()
            self.last_save = now

    def final_push(self):
        if len(self.results) > 0:
            self._push_to_database()
        if self.local is not None:
            self.local.dispose()
        self.local = None
        if self.remote is not None:
            self.remote.dispose()
        self.remote = None
        self.credentials = None
        self.project_id = None

    def _first_result(self) -> DataFrame | None:
        return self.results[0] if len(self.results) > 0 else None

    def push_batch(self):
        now = _now()
        df = self._first_result()
        if df is not None and (self._df_size_check(df) or (now - self.last_save) > timedelta(seconds=60.0)):
            self._push_to_database()
            self.last_save = now

    def batch_result(self, result: DataFrame):
        self.results.append(result)
        if self._df_size_check(result):  # If the batch write is already large, push it.
            logger.warning(f'batch_result(): Early Push: Number of Columns: {result.shape[1]}; ' +
                           f'Length of DataFrames: {sum(len(df) for df in self.results)}')
            self.push_batch()

    def read_table(self) -> DataFrame:
        df = None
        if self.table_name is not None:
            if self.remote is not None:
                try:
                    df = pd.read_sql_query(f'SELECT * FROM {self.table_name}', self.remote)
                except (ValueError, OperationalError) as e:
                    logger.error(f'{e}')
                    df = None
            elif self.credentials is not None:
                try:
                    df = pandas_gbq.read_gbq(f'SELECT * FROM `EMS.{self.table_name}`',
                                             credentials=self.credentials, progress_bar_type=None)
                except pandas_gbq.exceptions.GenericGBQException as e:
                    logger.error(f'{e}')
                    df = None
            elif self.project_id is not None:
                try:
                    df = pandas_gbq.read_gbq(f'SELECT * FROM `EMS.{self.table_name}`',
                                             project_id=self.project_id, progress_bar_type=None)
                except pandas_gbq.exceptions.GenericGBQException as e:
                    logger.error(f'{e}')
                    df = None
            elif self.local is not None:
                try:
                    df = pd.read_sql_table(self.table_name, self.local)
                except ValueError:
                    df = None
        return df

    def read_params(self, params: list) -> DataFrame:
        df = None
        if self.table_name is not None:
            if len(params) > 0:
                keys = ','.join(sorted(params[0].keys()))
                if self.remote is not None:
                    try:
                        df = pd.read_sql_query(f'SELECT DISTINCT {keys} FROM {self.table_name}', self.remote)
                    except (ValueError, OperationalError) as e:
                        logger.error(f'{e}')
                        df = None
                elif self.credentials is not None:
                    try:
                        df = pandas_gbq.read_gbq(f'SELECT DISTINCT {keys} FROM `EMS.{self.table_name}`',
                                                 credentials=self.credentials, progress_bar_type=None)
                    except pandas_gbq.exceptions.GenericGBQException as e:
                        logger.error(f'{e}')
                        df = None
                elif self.project_id is not None:
                    try:
                        df = pandas_gbq.read_gbq(f'SELECT DISTINCT {keys} FROM `EMS.{self.table_name}`',
                                                 project_id=self.project_id, progress_bar_type=None)
                    except pandas_gbq.exceptions.GenericGBQException as e:
                        logger.error(f'{e}')
                        df = None
                elif self.local is not None:
                    try:
                        df = pd.read_sql_query(f'SELECT DISTINCT {keys} FROM {self.table_name}', self.local)
                    except (ValueError, OperationalError) as e:
                        logger.error(f'{e}')
                        df = None
            else:
                df = self.read_table()
        return df


# The Cloud SQL Python Connector can be used along with SQLAlchemy using the
# 'creator' argument to 'create_engine'
def create_remote_connection_engine() -> Engine:
    def get_conn() -> Connection:
        connector = Connector()
        connection: Connection = connector.connect(
            os.environ["POSTGRES_CONNECTION_NAME"],
            "pg8000",
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASS"],
            db=os.environ["POSTGRES_DB"],
        )
        return connection

    engine = create_engine(
        "postgresql+pg8000://",
        creator=get_conn,
        echo=False,
        pool_pre_ping=True  # Force reestablishing the connection.
    )
    engine.dialect.description_encoding = None
    return engine


def active_remote_engine() -> (Engine, MetaData):
    remote = create_remote_connection_engine()
    metadata = MetaData()
    try:
        metadata.reflect(remote)  # Causes a DB query.
        return remote, metadata
    except SQLAlchemyError as e:
        logger.error("%s", e)
        remote.dispose()
    return None, None


def get_gbq_credentials(cred_name: str = 'hs-deep-lab-donoho-3d5cf4ffa2f7.json') -> service_account.Credentials:
    path = f'~/.config/gcloud/{cred_name}'  # Pandas-GBQ-DataSource
    expanded_path = os.path.expanduser(path)
    credentials = service_account.Credentials.from_service_account_file(expanded_path)
    return credentials


class EvalOnCluster(object):

    def __init__(self, client: Client,
                 table_name: str = None, credentials: service_account.credentials = None):
        self.db = Databases(table_name, None, credentials, None)
        self.client = client
        self.credentials = credentials
        self.computations = None  # Iterable returning (future, df).
        self.keys = None

    def key_from_params(self, params: dict) -> tuple:
        if self.keys is None:
            self.keys = sorted(params.keys())
        return tuple(params[k] for k in self.keys)

    def eval_params(self, instance: callable, params: dict) -> tuple:
        """
        Evaluate the instance with the params and return a tuple of param values that could become a key in a dict.
        :param instance: The `callable` to be invoked on the cluster.
        :param params: The `kwargs` to be passed to the `instance`
        :return: A tuple of param values suitable to become a key in a dict.
        """

        futures = self.client.map(lambda p: instance(**p), [params])  # To isolate kwargs, use a lambda function.
        if self.computations is None:
            self.computations = as_completed(futures, with_results=True)
        else:
            self.computations.update(futures)
        return self.key_from_params(params)

    def __iter__(self):
        return self

    def __next__(self):
        future, result = self.computations.__next__()
        self.db.push(result)
        future.release()  # EP function; release the data; will not be reused.
        values = result[self.keys].to_numpy()
        return result, tuple(v for v in values[0])

    def __aiter__(self):
        return self

    async def __anext__(self):
        future, result = await self.computations.__anext__()
        self.db.push(result)
        future.release()  # EP function; release the data; will not be reused.
        values = result[self.keys].to_numpy()
        return result, tuple(v for v in values[0])

    def result(self) -> (DataFrame, tuple):  # Return a DataFrame and a key.
        future, result = next(self.computations)
        self.db.push(result)
        future.release()  # EP function; release the data; will not be reused.
        values = result[self.keys].to_numpy()
        yield result, tuple(v for v in values[0])

    def final_push(self):
        self.db.final_push()
        self.client.shutdown()


def on_worker() -> bool:
    import distributed.worker

    try:
        _ = distributed.worker.get_worker()
        return True
    except ValueError:
        return False


def get_dataset(key: str) -> DataFrame:
    if on_worker():
        with worker_client() as wc:
            df = wc.get_dataset(name=key, default=None)
    else:
        wc = Client.current(allow_global=True)
        df = wc.get_dataset(name=key, default=None)
    return df


def unroll_parameters(parameters: dict) -> list:
    """
    'parameters': {
        'm': [50],
        'n': [1275, 2550, 3825],
        'mc': list(range(50)),
        'c4': linspace(0.25, 2.5, 10),
        'p': concatenate((linspace(0.02, 0.10, 9), linspace(0.15, 0.50, 8))),
        'q_type': [21],
        'd_type': [3]
        }
    """
    unrolled = []
    for key, values in parameters.items():
        next_unroll = []
        for value in values:
            roll = copy.deepcopy(unrolled)
            if len(roll) > 0:
                for param in roll:
                    param[key] = value
            else:
                roll.append({key: value})
            next_unroll.extend(roll)
        unrolled = next_unroll
    return unrolled


def unroll_parameters_gpt(parameters: dict) -> list:
    """
    parameters = {
        'm': [50],
        'n': [1275, 2550, 3825],
        'mc': list(range(50)),
        'c4': np.linspace(0.25, 2.5, 10),
        'p': np.linspace(0.02, 0.10, 9),
        'q_type': [21],
        'd_type': [3]
        }
    """
    unrolled = []
    import itertools

    # Get all possible combinations of values from the lists
    combinations = list(itertools.product(*parameters.values()))

    # Create dictionaries with keys and one combination of values
    for combo in combinations:
        combined = {key: value for key, value in zip(parameters.keys(), combo)}
        unrolled.append(combined)
    return unrolled


def update_index(index: int, df: DataFrame) -> DataFrame:
    as_list = df.index.tolist()
    for i in range(len(as_list)):
        as_list[i] = index + i
    df.index = as_list
    return df


def remove_stop_list(unrolled: list, stop: list) -> list:
    result = []
    sl = stop.copy()  # Copy the stop_list to allow it to shrink as items are found and removed.
    for param in unrolled:
        for s_param in sl:
            if len(param) == len(s_param):  # TODO: What should we do in the case of mismatched lengths?
                for k, v in s_param.items():
                    if param[k] != v:
                        break
                else:  # no_break => all (k, v) are equal. param is IN the stop_list.
                    sl.remove(s_param)
                    break
        else:  # no_break => param is NOT in the stop_list.
            result.append(param)
    return result


def timestamp() -> int:
    return floor(_now().timestamp())


def write_json(d: dict, fn: str):
    with open(fn, 'w') as json_file:
        json.dump(d, json_file, indent=4)


def read_json(fn: str) -> dict:
    with open(fn, 'r') as json_file:
        d = json.load(json_file)
    return d


def record_experiment(experiment: dict):
    table_name = experiment.get('table_name', None)
    if table_name is not None:
        now_ts = timestamp()
        write_json(experiment, table_name + f'-{now_ts}.json')


def unroll_experiment(experiment: dict) -> list:
    parameters = []
    if params := experiment.get('params', None):
        for p in params:
            parameters.extend(unroll_parameters_gpt(p))
    elif multi_res := experiment.get('multi_res', None):
        for p in multi_res:
            parameters.extend(unroll_parameters_gpt(p))
    elif params := experiment.get('parameters', None):
        parameters = unroll_parameters_gpt(params)
    if stop_list := experiment.get('stop_list', None):
        parameters = remove_stop_list(parameters, stop_list)
    return parameters


def dedup_experiment(df: DataFrame, params: list) -> list:
    dedup = []
    if len(params) > 0:
        keys = sorted(params[0].keys())
        df_values = set(tuple(row) for row in df[keys].to_numpy())

        for p in params:
            values = tuple(p[k] for k in keys)
            if values not in df_values:
                dedup.append(p)
                df_values.add(values)
    return dedup


def do_test_experiment(experiment: dict, instance: callable, client: Client,
                       remote: Engine = None,
                       credentials: service_account.credentials = None, project_id: str = None):
    # Read the DB level parameters.
    table_name = experiment.get('table_name', None)
    db = Databases(table_name, remote, credentials, project_id)

    # Save the experiment domain.
    record_experiment(experiment)

    # Prepare parameters.
    parameters = unroll_experiment(experiment)
    df = db.read_params(parameters)
    if df is not None and len(df.index) > 0:
        parameters = dedup_experiment(df, parameters)
    df = None  # Free up the DataFrame.
    random.shuffle(parameters)
    instance_count = len(parameters)
    logger.info(f'Number of Instances to calculate: {instance_count}')


def dedup_experiment_from_db(experiment: dict, remote: Engine = None,
                             credentials: service_account.credentials = None) -> list:
    # Read the DB level parameters.
    table_name = experiment.get('table_name', None)
    db = Databases(table_name, remote, credentials)

    # Prepare parameters.
    parameters = unroll_experiment(experiment)
    df = db.read_params(parameters)
    if df is not None and len(df.index) > 0:
        parameters = dedup_experiment(df, parameters)
    return parameters  # Return the parameters not yet calculated.


def do_experiment(instance: callable, parameters: list, db: Databases, client: Client):
    instance_count = len(parameters)
    i = 0
    logger.info(f'Number of Instances to calculate: {instance_count}')
    # Start the computation.
    tick = time.perf_counter()
    futures = client.map(lambda p: instance(**p), parameters, batch_size=BATCH_SIZE)
    for batch in as_completed(futures, with_results=True).batches():
        for future, result in batch:
            i += 1
            if not (i % 10):  # Log results every tenth output
                tock = time.perf_counter() - tick
                remaining_count = instance_count - i
                s_i = tock / i
                logger.info(f'Count: {i}; Time: {round(tock)}; Seconds/Instance: {s_i:0.4f}; ' +
                            f'Remaining (s): {round(remaining_count * s_i)}; Remaining Count: {remaining_count}')
                logger.info(result)
            db.batch_result(result)
            future.release()  # As these are Embarrassingly Parallel tasks, clean up memory.
        db.push_batch()
    db.final_push()
    total_time = time.perf_counter() - tick
    logger.info(f"Performed experiment in {total_time:0.4f} seconds")
    if instance_count > 0:
        logger.info(f"Count: {instance_count}, Seconds/Instance: {(total_time / instance_count):0.4f}")


def do_on_cluster(experiment: dict, instance: callable, client: Client,
                  remote: Engine = None,
                  credentials: service_account.credentials = None, project_id: str = None):
    logger.info(f'{client}')
    # Read the DB level parameters.
    table_name = experiment.get('table_name', None)
    db = Databases(table_name, remote, credentials, project_id)

    # Save the experiment domain.
    record_experiment(experiment)

    # Prepare parameters.
    parameters = unroll_experiment(experiment)
    df = db.read_params(parameters)
    if df is not None and len(df.index) > 0:
        parameters = dedup_experiment(df, parameters)
    df = None  # Free up the DataFrame.
    if len(parameters) > 0:
        random.shuffle(parameters)
        do_experiment(instance, parameters, db, client)
    else:
        logger.warning('Database is complete.')
    client.shutdown()


if __name__ == '__main__':
    _touch_db_url(DB_URL)
    # d = {
    #     'm': [50],
    #     'n': [1275, 2550, 3825],
    #     'mc': list(range(50)),
    #     'c4': np.linspace(0.25, 2.5, 10),
    #     'p': np.linspace(0.02, 0.10, 9),
    #     'q_type': [21],
    #     'd_type': [3]
    #     }
    # p = unroll_parameters_gpt(d)
    # print(p)
