import tarfile
from ibm_watson_machine_learning import APIClient
import time, json, base64
import pandas as pd
from dotenv import load_dotenv
import os
import unicodedata

def run():
    #load_dotenv()
    WML_API_KEY = "kI0O5Y3O037SXJ3_1wiDBKFABLKAIrEaAsicdjI0Hu0t"
    SPACE_ID = "5b4f04fa-0a13-4793-8922-e0228341aa72"
    WML_URL = "https://us-south.ml.cloud.ibm.com"


    files_to_add = [
        "calendar.json",
        "costs.json",
        "demand.json",
        "fields_flags.json",
        "params.json",
        "teams.json"
    ]


    tar = tarfile.open("modelo.tar.gz", "w:gz")
    def reset(tarinfo):
        tarinfo.uid = tarinfo.gid = 0
        tarinfo.uname = tarinfo.gname = "root"
        return tarinfo

    tar.add("./scripts/model.py", arcname="model.py", filter=reset)
    for file in files_to_add:
        tar.add(f"./inputs/{file}", arcname=file, filter=reset)
    tar.close()

    wml_credentials = {
        "apikey": WML_API_KEY,
        "url": WML_URL
    }

    client = APIClient(wml_credentials)
    client.set.default_space(SPACE_ID)

    repository_model_spec = client.software_specifications.get_id_by_name("do_20.1")
    mnist_metadata = {
        client.repository.ModelMetaNames.NAME: "transportistas-opt",
        client.repository.ModelMetaNames.TYPE: "do-docplex_20.1",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: repository_model_spec,
    }
    model_details = client.repository.store_model(model='modelo.tar.gz', meta_props=mnist_metadata)
    model_uid = client.repository.get_model_id(model_details)

    meta_props = {
        client.deployments.ConfigurationMetaNames.NAME: "Modelo Transportistas Deployment",
        client.deployments.ConfigurationMetaNames.DESCRIPTION: "Modelo de cerezas",
        client.deployments.ConfigurationMetaNames.BATCH: {},
        client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {'name': 'S', 'num_nodes': 1}
    }

    deployment_details = client.deployments.create(model_uid, meta_props=meta_props)
    deployment_uid = client.deployments.get_id(deployment_details)

    solve_payload = {
        "solve_parameters": {
            "oaas.logAttachmentName": "log.txt",
            "oaas.logTailEnabled": "true",
            "oaas.resultsFormat": "JSON"   # <— importante para tener solution.json
        },
        client.deployments.DecisionOptimizationMetaNames.INPUT_DATA: [
            # (si tuvieras input_data embebidos, van aquí)
        ],
        client.deployments.DecisionOptimizationMetaNames.OUTPUT_DATA: [
            {"id": "solution.json"},
            {"id": "log.txt"}              # <— pedimos explícitamente el log
        ]
    }


    job_details = client.deployments.create_job(deployment_uid, solve_payload)
    job_uid = client.deployments.get_job_uid(job_details)

    while job_details['entity']['decision_optimization']['status']['state'] not in ['completed', 'failed', 'canceled']:
        print(job_details['entity']['decision_optimization']['status']['state'] + '...')
        time.sleep(5)
        job_details = client.deployments.get_job_details(job_uid)

    solve_status = job_details['entity']['decision_optimization']['solve_state']['solve_status']
    print(f"Estado del trabajo: {solve_status}")

    solution_json = None
    execution_log = None

    try:
        outputs = job_details['entity']['decision_optimization'].get('output_data', [])
        for output in outputs:
            oid = output.get('id', '')
            # Los contenidos vuelven en base64
            content_b64 = output.get('content')
            if not content_b64:
                continue
            raw = base64.b64decode(content_b64)

            if oid == 'solution.json':
                solution_text = raw.decode('utf-8', errors='replace')
                solution_json = json.loads(solution_text)
                with open('solucion.json', 'w', encoding='utf-8') as f:
                    f.write(json.dumps(solution_json, indent=4, ensure_ascii=False))
                print("Solución guardada en 'solucion.json'")

            elif oid == 'log.txt':
                execution_log = raw.decode('utf-8', errors='replace')
                with open('log.txt', 'w', encoding='utf-8') as f:
                    f.write(execution_log)
                print("Log de ejecución guardado en 'log.txt'")

    except Exception as e:
        print(f"Error al obtener outputs del job: {e}")

    if solution_json is None:
        print(f"El trabajo terminó con estado: {solve_status} y no se encontró 'solution.json'.")


  
