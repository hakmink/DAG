from airflow.operators.python_operator import PythonOperator
from airflow.models import DAG
from datetime import datetime
from hcc.grant.datasets.data_loader import zip_file_loader
from hcc.grant.preprocessing.wrangling import prepare_data
from hcc.grant.modeling.gbdt_lr import gbdt_lr_model
from hcc.grant.evaluation.evaluator import evaluator
from hcc.grant.serving.save_result import insert_postgres_db

args = {
    'owner': 'airflow',
    'start_date': datetime(2018, 3, 5, 0),
    'provide_context': True
}

dag = DAG(
    dag_id='alab_gbdt_lr',
    default_args=args,
    schedule_interval='@once')


def load_data(**kwargs):
    return zip_file_loader()


def preprocessing(**kwargs):
    ti = kwargs['ti']
    raw_dataset = ti.xcom_pull(key=None, task_ids='load_data')

    datasets_dir = raw_dataset["datasets_dir"]
    traning_data = "/".join([datasets_dir, raw_dataset['training']])
    test_data = "/".join([datasets_dir, raw_dataset['test']])
    return prepare_data(traning_data, test_data)


def modeling(**kwargs):
    ti = kwargs['ti']
    preprocessed_dataset = ti.xcom_pull(key=None, task_ids='preprocessing')
    return gbdt_lr_model(preprocessed_dataset)


def evalutaion(**kwargs):
    ti = kwargs['ti']
    preprocessed_dataset = ti.xcom_pull(key=None, task_ids='preprocessing')
    modeling_result_dataset = ti.xcom_pull(key=None, task_ids='modeling')
    return evaluator(modeling_result_dataset, preprocessed_dataset)


def save_result(**kwargs):
    ti = kwargs['ti']
    evaluated_dataset = ti.xcom_pull(key=None, task_ids='evalutaion')
    return insert_postgres_db(evaluated_dataset)


node_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag)

node_preprocessing = PythonOperator(
    task_id='preprocessing',
    python_callable=preprocessing,
    dag=dag)

node_modeling = PythonOperator(
    task_id='modeling',
    python_callable=modeling,
    dag=dag)

node_evalutaion = PythonOperator(
    task_id='evalutaion',
    python_callable=evalutaion,
    dag=dag)

node_save_result = PythonOperator(
    task_id='save_result',
    python_callable=save_result,
    dag=dag)

node_load_data >> node_preprocessing >> node_modeling >> node_evalutaion >> node_save_resultnet
