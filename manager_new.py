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
from math import floor
from pathlib import Path

import pandas as pd
from pandas import DataFrame
import pandas_gbq
import pandas_gbq.exceptions
from dask.distributed import Client, as_completed
from google.cloud.sql.connector import Connector
from google.oauth2 import service_account
from pg8000.dbapi import Connection
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.schema import MetaData

BATCH_SIZE = 4096
NUM_CELLS = 200 * 1000  # 200 rows x 1,000 columns
logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _touch_db_url(db_url: str):
    parts = db_url.split('sqlite:///')
    if len(parts) == 2:
        p = Path(parts[1])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)


class Databases:

    def __init__(
        self,
        table_name: str,
        remote: Engine = None,
        credentials: service_account.Credentials = None,
        project_id: str = None
    ):
        self.results = []
        self.last_save = _now()
        self.last_gbq_push = self.last_save
        self.table_name = table_name

        db_url = 'sqlite:///data/EMS.db3'
        _touch_db_url(db_url)
        self.local = create_engine(db_url, echo=False)
        self.remote = remote
        self.credentials = credentials
        self.project_id = project_id

    def _push_to_database(self):
        # Early exit if no new results
        if not self.results:
            return

        # 1) Prepare DataFrame
        df = pd.concat(self.results, ignore_index=True)
        rows, cols = df.shape
        logger.warning(f'_push_to_database(): Appending {rows} rows and {cols} cols to local DB.')

        # 2) Write locally
        with self.local.connect() as ldb:
            # df.to_sql(self.table_name, ldb, if_exists='append', method='multi')
            for start in range(0, len(df), 28):
                df.iloc[start:start + 28].to_sql(
                    self.table_name,
                    ldb,
                    if_exists='append',
                    method='multi',
                    index=False
                )

        # 3) Throttle streaming inserts (only if GBQ configured)
        now = _now()
        if (self.credentials or self.project_id) and (now - self.last_gbq_push) >= timedelta(minutes=10):
            try:
                # stream the entire accumulated batch
                df_to_push = pd.concat(self.results, ignore_index=True)
                count, _ = df_to_push.shape
                df_to_push.to_gbq(
                    f'EMS.{self.table_name}',
                    if_exists='append',
                    progress_bar=False,
                    credentials=self.credentials,
                    project_id=self.project_id
                )
                self.last_gbq_push = now
                logger.info(f'Flushed {rows} rows to GBQ EMS.{self.table_name} at {now}')
                self.results.clear()
            except Exception as e:
                logger.error("GBQ streaming insert failed: %s", e)

    def push(self, result: DataFrame):
        now = _now()
        self.results.append(result)
        # trigger push by size or time
        if (len(self.results) > 1 and self._df_size_check(result)) or ((now - self.last_save) > timedelta(seconds=60)):
            self._push_to_database()
            self.last_save = now

    def batch_result(self, result: DataFrame):
        self.results.append(result)
        if self._df_size_check(result):
            self._push_to_database()
            self.last_save = _now()

    def push_batch(self):
        """
        Alias for legacy push_batch calls: flush buffer as in _push_to_database.
        """
        self._push_to_database()
        self.last_save = _now()

    def final_push(self):
        # Early exit if no pending results
        if not self.results:
            self._cleanup_engines()
            return

        # 4) Final flush
        df = pd.concat(self.results, ignore_index=True)
        rows, cols = df.shape
        logger.warning(f'final_push(): Appending remaining {rows} rows and {cols} cols to local DB.')

        with self.local.connect() as ldb:
            # df.to_sql(self.table_name, ldb, if_exists='append', method='multi')
            for start in range(0, len(df), 28):
                df.iloc[start:start + 28].to_sql(
                    self.table_name,
                    ldb,
                    if_exists='append',
                    method='multi',
                    index=False
                )

        # 5) Final streaming insert
        if self.credentials or self.project_id:
            try:
                df.to_gbq(
                    f'EMS.{self.table_name}',
                    if_exists='append',
                    progress_bar=False,
                    credentials=self.credentials,
                    project_id=self.project_id
                )
                # record final push time
                self.last_gbq_push = _now()
            except Exception as e:
                logger.error("Final GBQ push failed: %s", e)

        self.results.clear()
        self._cleanup_engines()

    def _cleanup_engines(self):
        self.local.dispose()
        self.local = None
        if self.remote:
            self.remote.dispose()
            self.remote = None
        self.credentials = None
        self.project_id = None

    def _df_size_check(self, df: DataFrame) -> bool:
        total_rows = sum(len(r) for r in self.results)
        return total_rows * df.shape[1] > NUM_CELLS

    def read_table(self) -> DataFrame:
        if self.remote:
            return None
        if self.credentials or self.project_id:
            try:
                return pd.read_gbq(
                    f'SELECT * FROM `EMS.{self.table_name}`',
                    credentials=self.credentials,
                    project_id=self.project_id
                )
            except pandas_gbq.exceptions.GenericGBQException as e:
                logger.error(e)
                return None
        try:
            return pd.read_sql_table(self.table_name, self.local)
        except Exception:
            return None

    def read_params(self, params: list) -> DataFrame:
        if not params:
            return self.read_table()
        keys = ','.join(sorted(params[0].keys()))
        query = f'SELECT DISTINCT {keys} FROM `EMS.{self.table_name}`'
        if self.credentials or self.project_id:
            try:
                return pd.read_gbq(
                    query,
                    credentials=self.credentials,
                    project_id=self.project_id
                )
            except pandas_gbq.exceptions.GenericGBQException as e:
                logger.error(e)
                return None
        try:
            return pd.read_sql_query(f'SELECT DISTINCT {keys} FROM {self.table_name}', self.local)
        except Exception:
            return None


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
    table_name = experiment['table_name']
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
    table_name = experiment['table_name']
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
    table_name = experiment['table_name']
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
    db_url = 'sqlite:///data/EMS.db3'
    _touch_db_url(db_url)
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
